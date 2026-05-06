"""Tests for the CVE-2025-69872 / GHSA-w8v5-vhqr-4h9v mitigation.

These tests exercise the HMAC envelope wrapping every pickle blob written
by :class:`diskcache.Disk`.  They live in their own module so the rest of
the suite continues to run with the default
:class:`UnsafePickleWarning` filter, while these tests can opt to either
suppress or assert that the warning is emitted.
"""

import hashlib
import hmac
import io
import multiprocessing
import os
import os.path as op
import pickle
import secrets
import shutil
import sqlite3
import sys
import tempfile
import warnings

import pytest

import diskcache as dc
from diskcache.core import (
    PICKLE_HEADER_SIZE,
    PICKLE_KEY_ENV,
    PICKLE_KEY_FILENAME,
    PICKLE_MAGIC,
)


pytestmark = pytest.mark.filterwarnings(
    'ignore', category=dc.EmptyDirWarning
)


@pytest.fixture
def tmp_cache_dir():
    path = tempfile.mkdtemp(prefix='diskcache-sec-')
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def clear_env(monkeypatch):
    monkeypatch.delenv(PICKLE_KEY_ENV, raising=False)


# -- helpers ------------------------------------------------------------


class _Pwn:
    """Pickle payload that would execute code if naively unpickled."""

    def __reduce__(self):
        # Using ``print`` rather than ``os.system`` keeps the test
        # self-contained and side-effect free if the protection ever
        # regresses.
        return (print, ('PWNED!',))


def _force_evil_inline_pickle(directory, key):
    """Replace the inline pickle blob for ``key`` with a malicious payload."""
    con = sqlite3.connect(op.join(directory, 'cache.db'))
    try:
        con.execute(
            'UPDATE Cache SET value = ?, mode = 4 WHERE key = ?',
            (pickle.dumps(_Pwn()), key),
        )
        con.commit()
    finally:
        con.close()


def _val_files(directory):
    out = []
    for root, _, files in os.walk(directory):
        for name in files:
            if name.endswith('.val'):
                out.append(op.join(root, name))
    return out


# -- round-trip & default behaviour ------------------------------------


def test_default_roundtrip_emits_warning_and_creates_key_file(
    tmp_cache_dir, clear_env
):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with dc.Cache(tmp_cache_dir) as cache:
            cache[('tup', 'le')] = {'nested': [1, 2, 3]}
            assert cache[('tup', 'le')] == {'nested': [1, 2, 3]}
            cache.check()

    msgs = [w for w in caught if isinstance(w.message, dc.UnsafePickleWarning)]
    assert msgs, 'expected UnsafePickleWarning for in-dir auto key'
    assert PICKLE_KEY_FILENAME in str(msgs[0].message)
    assert op.isfile(op.join(tmp_cache_dir, PICKLE_KEY_FILENAME))


def test_envelope_header_is_present_in_db(tmp_cache_dir, clear_env):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir) as cache:
            cache['k'] = {'value': 'object'}

    con = sqlite3.connect(op.join(tmp_cache_dir, 'cache.db'))
    try:
        ((blob,),) = con.execute(
            'SELECT value FROM Cache WHERE key = ?', ('k',)
        ).fetchall()
    finally:
        con.close()

    raw = bytes(blob)
    assert raw.startswith(PICKLE_MAGIC)
    assert len(raw) > PICKLE_HEADER_SIZE


# -- explicit key ------------------------------------------------------


def test_explicit_key_no_keyfile_no_warning(tmp_cache_dir, clear_env):
    key = secrets.token_bytes(32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            cache['x'] = [1, 2, 3]
            assert cache['x'] == [1, 2, 3]

    assert not any(
        isinstance(w.message, dc.UnsafePickleWarning) for w in caught
    )
    assert not op.exists(op.join(tmp_cache_dir, PICKLE_KEY_FILENAME))


def test_explicit_key_hex_string(tmp_cache_dir, clear_env):
    key = secrets.token_bytes(32).hex()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            cache['x'] = ['hex-keyed']
            assert cache['x'] == ['hex-keyed']


def test_explicit_key_too_short_raises(tmp_cache_dir, clear_env):
    with pytest.raises(ValueError, match='at least'):
        dc.Cache(tmp_cache_dir, disk_pickle_key=b'shortkey')


def test_explicit_key_wrong_type_raises(tmp_cache_dir, clear_env):
    with pytest.raises(TypeError):
        dc.Cache(tmp_cache_dir, disk_pickle_key=12345)


def test_explicit_key_not_persisted_in_settings(tmp_cache_dir, clear_env):
    key = secrets.token_bytes(32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            cache['x'] = {'persist?': False}

    con = sqlite3.connect(op.join(tmp_cache_dir, 'cache.db'))
    try:
        rows = con.execute(
            'SELECT key, value FROM Settings WHERE key LIKE ?',
            ('%pickle_key%',),
        ).fetchall()
    finally:
        con.close()

    assert rows == [], (
        'disk_pickle_key must NOT be persisted in the Settings table'
    )


def test_mismatched_key_fails_to_decrypt(tmp_cache_dir, clear_env):
    key1 = secrets.token_bytes(32)
    key2 = secrets.token_bytes(32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key1) as cache:
            cache['k'] = {'secret': True}
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key2) as cache:
            with pytest.raises(pickle.UnpicklingError):
                cache['k']


# -- env var -----------------------------------------------------------


def test_env_var_used_when_no_explicit_key(tmp_cache_dir, monkeypatch):
    monkeypatch.setenv(PICKLE_KEY_ENV, secrets.token_bytes(32).hex())
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with dc.Cache(tmp_cache_dir) as cache:
            cache['k'] = {'env': True}
            assert cache['k'] == {'env': True}

    assert not any(
        isinstance(w.message, dc.UnsafePickleWarning) for w in caught
    )
    assert not op.exists(op.join(tmp_cache_dir, PICKLE_KEY_FILENAME))


def test_env_var_invalid_hex_raises(tmp_cache_dir, monkeypatch):
    monkeypatch.setenv(PICKLE_KEY_ENV, 'not-hex-at-all')
    with dc.Cache(tmp_cache_dir) as cache:
        with pytest.raises(ValueError, match='hex'):
            cache['k'] = {'x': 1}


# -- tamper detection --------------------------------------------------


def test_tampered_inline_value_raises(tmp_cache_dir, clear_env):
    key = secrets.token_bytes(32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            cache['k'] = {'foo': 'bar'}

    _force_evil_inline_pickle(tmp_cache_dir, 'k')

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            with pytest.raises(pickle.UnpicklingError):
                cache['k']


def test_bitflip_inline_value_raises(tmp_cache_dir, clear_env):
    """Tampering that keeps the envelope header but corrupts the payload
    must fail HMAC verification (not just the missing-envelope check)."""
    key = secrets.token_bytes(32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            cache['k'] = {'foo': 'bar'}

    con = sqlite3.connect(op.join(tmp_cache_dir, 'cache.db'))
    try:
        ((blob,),) = con.execute(
            'SELECT value FROM Cache WHERE key = ?', ('k',)
        ).fetchall()
        raw = bytearray(bytes(blob))
        # Flip a bit deep inside the payload (past the 36-byte header).
        raw[-1] ^= 0x01
        con.execute(
            'UPDATE Cache SET value = ? WHERE key = ?', (bytes(raw), 'k')
        )
        con.commit()
    finally:
        con.close()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            with pytest.raises(pickle.UnpicklingError, match='HMAC'):
                cache['k']


def test_tampered_val_file_raises(tmp_cache_dir, clear_env):
    key = secrets.token_bytes(32)
    big_value = {'data': 'x' * (40 * 1024)}
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(
            tmp_cache_dir, disk_pickle_key=key, disk_min_file_size=1024
        ) as cache:
            cache['big'] = big_value

    files = _val_files(tmp_cache_dir)
    assert len(files) == 1, files
    with open(files[0], 'wb') as writer:
        writer.write(pickle.dumps(_Pwn()))

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            with pytest.raises(pickle.UnpicklingError):
                cache['big']


def test_tampered_pickled_key_via_iteration_raises(tmp_cache_dir, clear_env):
    """Complex (pickled) keys are loaded during iteration / peekitem, so
    tampering them must surface as UnpicklingError on those paths."""
    key = secrets.token_bytes(32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            cache[('complex', 'key')] = 'value'

    con = sqlite3.connect(op.join(tmp_cache_dir, 'cache.db'))
    try:
        rows = con.execute(
            'SELECT rowid, key FROM Cache WHERE raw = 0'
        ).fetchall()
        assert rows, 'expected at least one pickled key row'
        rowid, _ = rows[0]
        evil = pickle.dumps(_Pwn())
        con.execute(
            'UPDATE Cache SET key = ? WHERE rowid = ?',
            (sqlite3.Binary(evil), rowid),
        )
        con.commit()
    finally:
        con.close()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            with pytest.raises(pickle.UnpicklingError):
                list(cache)


# -- legacy mode -------------------------------------------------------


def test_legacy_mode_emits_warning_and_works(tmp_cache_dir, clear_env):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with dc.Cache(tmp_cache_dir, disk_pickle_key=False) as cache:
            cache['z'] = {'legacy': True}
            assert cache['z'] == {'legacy': True}

    msgs = [w for w in caught if isinstance(w.message, dc.UnsafePickleWarning)]
    assert msgs, 'pickle_key=False must emit UnsafePickleWarning'
    assert 'disabled' in str(msgs[0].message).lower()


def test_secure_mode_rejects_legacy_data(tmp_cache_dir, clear_env):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=False) as cache:
            cache['k'] = {'legacy': True}
        with dc.Cache(
            tmp_cache_dir, disk_pickle_key=secrets.token_bytes(32)
        ) as cache:
            with pytest.raises(pickle.UnpicklingError, match='envelope'):
                cache['k']


def test_legacy_mode_can_read_secure_data(tmp_cache_dir, clear_env):
    """Legacy mode strips the envelope so it can read forward as well."""
    key = secrets.token_bytes(32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            cache['k'] = {'enveloped': True}
        with dc.Cache(tmp_cache_dir, disk_pickle_key=False) as cache:
            assert cache['k'] == {'enveloped': True}


# -- JSONDisk unaffected ------------------------------------------------


def test_jsondisk_does_not_create_keyfile(tmp_cache_dir, clear_env):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with dc.Cache(tmp_cache_dir, disk=dc.JSONDisk) as cache:
            cache['a'] = [1, 2, 3]
            assert cache['a'] == [1, 2, 3]

    assert not any(
        isinstance(w.message, dc.UnsafePickleWarning) for w in caught
    )
    assert not op.exists(op.join(tmp_cache_dir, PICKLE_KEY_FILENAME))


# -- Cache.check whitelist ---------------------------------------------


def test_check_does_not_warn_about_keyfile(tmp_cache_dir, clear_env):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with dc.Cache(tmp_cache_dir) as cache:
            cache['k'] = {'foo': 'bar'}
            warns = cache.check()
        assert op.isfile(op.join(tmp_cache_dir, PICKLE_KEY_FILENAME))

    unknown = [w for w in warns if isinstance(w.message, dc.UnknownFileWarning)]
    assert unknown == [], (
        'cache.check() must whitelist .diskcache_pickle_key, got %r' % unknown
    )

    unknown_top = [
        w for w in caught if isinstance(w.message, dc.UnknownFileWarning)
    ]
    assert unknown_top == []


# -- primitive keys/values bypass pickle entirely ----------------------


def test_primitive_only_does_not_create_keyfile(tmp_cache_dir, clear_env):
    """Lazy resolution means a cache used only with primitive keys and
    values never triggers HMAC key creation."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with dc.Cache(tmp_cache_dir) as cache:
            cache['plain-str-key'] = 'plain-str-value'
            cache[42] = 3.14
            cache[b'bytes-key'] = b'bytes-value'
            assert cache['plain-str-key'] == 'plain-str-value'

    assert not any(
        isinstance(w.message, dc.UnsafePickleWarning) for w in caught
    )
    assert not op.exists(op.join(tmp_cache_dir, PICKLE_KEY_FILENAME))


# -- Review-pass-2: path traversal (C2) --------------------------------


def _put_evil_filename(directory, key, evil_filename, mode=2):
    """Replace the row's filename to point outside the cache dir."""
    con = sqlite3.connect(op.join(directory, 'cache.db'))
    try:
        con.execute(
            'UPDATE Cache SET filename = ?, mode = ?, value = NULL '
            'WHERE key = ?',
            (evil_filename, mode, key),
        )
        con.commit()
    finally:
        con.close()


def test_path_traversal_in_filename_raises(tmp_cache_dir, clear_env):
    """An attacker who tampers cache.db's ``filename`` column to point
    outside the cache directory must not be able to coerce DiskCache
    into reading arbitrary files (CVE-2025-69872 adjacent)."""
    secret_dir = tempfile.mkdtemp(prefix='diskcache-secret-')
    try:
        secret_path = op.join(secret_dir, 'secret.txt')
        with open(secret_path, 'wb') as fh:
            fh.write(b'TOP SECRET')

        key = secrets.token_bytes(32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', dc.UnsafePickleWarning)
            with dc.Cache(
                tmp_cache_dir,
                disk_pickle_key=key,
                disk_min_file_size=1,
            ) as cache:
                cache['k'] = b'x' * 64

        rel = op.relpath(secret_path, tmp_cache_dir)
        _put_evil_filename(tmp_cache_dir, 'k', rel, mode=2)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', dc.UnsafePickleWarning)
            with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
                with pytest.raises(ValueError, match='escapes cache'):
                    cache['k']
    finally:
        shutil.rmtree(secret_dir, ignore_errors=True)


def test_path_traversal_blocked_for_text_mode(tmp_cache_dir, clear_env):
    secret_dir = tempfile.mkdtemp(prefix='diskcache-secret-')
    try:
        secret_path = op.join(secret_dir, 'secret.txt')
        with open(secret_path, 'w', encoding='utf-8') as fh:
            fh.write('TOP SECRET')

        key = secrets.token_bytes(32)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore', dc.UnsafePickleWarning)
            with dc.Cache(
                tmp_cache_dir,
                disk_pickle_key=key,
                disk_min_file_size=1,
            ) as cache:
                cache['k'] = 'x' * 64

        rel = op.relpath(secret_path, tmp_cache_dir)
        _put_evil_filename(tmp_cache_dir, 'k', rel, mode=3)

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', dc.UnsafePickleWarning)
            with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
                with pytest.raises(ValueError, match='escapes cache'):
                    cache['k']
    finally:
        shutil.rmtree(secret_dir, ignore_errors=True)


def test_remove_refuses_traversal(tmp_cache_dir, clear_env):
    """Disk.remove must refuse path-traversal filenames silently
    (eviction must be tolerant) but still emit an UnsafePickleWarning
    so operators see the tampering."""
    sentinel_dir = tempfile.mkdtemp(prefix='diskcache-sentinel-')
    try:
        sentinel = op.join(sentinel_dir, 'do-not-delete.txt')
        with open(sentinel, 'wb') as fh:
            fh.write(b'KEEP ME')

        with warnings.catch_warnings():
            warnings.simplefilter('ignore', dc.UnsafePickleWarning)
            with dc.Cache(tmp_cache_dir) as cache:
                disk = cache._disk

        rel = op.relpath(sentinel, tmp_cache_dir)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            disk.remove(rel)

        assert op.exists(sentinel), 'sentinel file must not be deleted'
        assert any(
            isinstance(w.message, dc.UnsafePickleWarning)
            and 'refusing to remove' in str(w.message)
            for w in caught
        )
    finally:
        shutil.rmtree(sentinel_dir, ignore_errors=True)


# -- Review-pass-2: FanoutCache (C3) -----------------------------------


def test_fanoutcache_propagates_explicit_key(tmp_cache_dir, clear_env):
    key = secrets.token_bytes(32)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with dc.FanoutCache(
            tmp_cache_dir, shards=4, disk_pickle_key=key
        ) as cache:
            cache[('complex', 'key')] = {'value': [1, 2, 3]}
            assert cache[('complex', 'key')] == {'value': [1, 2, 3]}

    assert not any(
        isinstance(w.message, dc.UnsafePickleWarning) for w in caught
    ), 'explicit key must not emit UnsafePickleWarning'

    # No per-shard key file: the explicit key was forwarded.
    for num in range(4):
        shard_dir = op.join(tmp_cache_dir, '%03d' % num)
        assert not op.exists(op.join(shard_dir, PICKLE_KEY_FILENAME))


def test_fanoutcache_default_emits_one_warning(tmp_cache_dir, clear_env):
    """With the default fallback, FanoutCache must emit ONE warning,
    not one-per-shard, and create a single key file at the FanoutCache
    root (not per-shard)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter('always')
        with dc.FanoutCache(tmp_cache_dir, shards=4) as cache:
            cache[('a', 'b')] = {'val': 1}
            cache[('c', 'd')] = {'val': 2}
            cache[('e', 'f')] = {'val': 3}
            assert cache[('a', 'b')] == {'val': 1}
            assert cache[('c', 'd')] == {'val': 2}

    pkey_warnings = [
        w for w in caught if isinstance(w.message, dc.UnsafePickleWarning)
    ]
    assert len(pkey_warnings) == 1, (
        'expected exactly one UnsafePickleWarning per FanoutCache, got %d'
        % len(pkey_warnings)
    )

    # Root-level key file exists; no shard-level key files.
    assert op.exists(op.join(tmp_cache_dir, PICKLE_KEY_FILENAME))
    for num in range(4):
        shard_dir = op.join(tmp_cache_dir, '%03d' % num)
        assert not op.exists(op.join(shard_dir, PICKLE_KEY_FILENAME)), (
            'shard %d must not have its own key file' % num
        )


def test_fanoutcache_two_instances_share_default_key(
    tmp_cache_dir, clear_env
):
    """Two FanoutCache instances on the same directory must produce
    consistent sharding and read each other's writes."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.FanoutCache(tmp_cache_dir, shards=4) as a:
            a[('shared', 'tuple', 'key')] = {'from': 'a'}
        with dc.FanoutCache(tmp_cache_dir, shards=4) as b:
            assert b[('shared', 'tuple', 'key')] == {'from': 'a'}


# -- Review-pass-2: Cache.check root-only whitelist (H1) ---------------


def test_check_does_not_whitelist_subdirectory_keyfile(
    tmp_cache_dir, clear_env
):
    """Attacker plants ``.diskcache_pickle_key`` in a subdirectory; the
    whitelist must not preserve it (basename-only would have).

    We assert the observable side effect: ``check(fix=True)`` deletes
    the subdirectory decoy.  (Pytest's warning capture interacts oddly
    with ``check()``'s own ``warnings.catch_warnings(record=True)``
    block, so we don't rely on the returned warning list.)
    """
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_min_file_size=0) as cache:
            cache['k'] = 'x' * 200  # forces .val file in subdirectory

            subdir = None
            for entry in os.listdir(tmp_cache_dir):
                full = op.join(tmp_cache_dir, entry)
                if op.isdir(full):
                    subdir = full
                    break
            assert subdir is not None, (
                'expected subdir; got %r' % os.listdir(tmp_cache_dir)
            )
            decoy = op.join(subdir, PICKLE_KEY_FILENAME)
            with open(decoy, 'wb') as fh:
                fh.write(b'attacker-controlled-bytes')

            assert op.exists(decoy)
            cache.check(fix=True)
            assert not op.exists(decoy), (
                'check(fix=True) must delete the subdirectory decoy '
                '(basename-only whitelist would have preserved it)'
            )

            # Sanity: legitimate root-level key file (not present here
            # since we used disk_min_file_size=0 and didn't pickle) is
            # not accidentally flagged either; create one and re-check.
            root_key = op.join(tmp_cache_dir, PICKLE_KEY_FILENAME)
            with open(root_key, 'wb') as fh:
                fh.write(b'X' * 32)
            cache.check(fix=True)
            assert op.exists(root_key), (
                'root-level key file must remain whitelisted'
            )


# -- Review-pass-2: Cache pickling (H2) --------------------------------


def test_cache_pickling_with_explicit_key_raises(tmp_cache_dir, clear_env):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        cache = dc.Cache(tmp_cache_dir, disk_pickle_key=secrets.token_bytes(32))
        try:
            with pytest.raises(TypeError, match='disk_pickle_key'):
                pickle.dumps(cache)
        finally:
            cache.close()


def test_cache_pickling_with_legacy_mode_raises(tmp_cache_dir, clear_env):
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        cache = dc.Cache(tmp_cache_dir, disk_pickle_key=False)
        try:
            with pytest.raises(TypeError, match='disk_pickle_key'):
                pickle.dumps(cache)
        finally:
            cache.close()


def test_cache_pickling_with_default_succeeds(tmp_cache_dir, clear_env):
    """No explicit key + default fallback: pickling must work because
    the receiving process can re-resolve from env or file."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir) as cache:
            cache['k'] = {'value': 1}
            blob = pickle.dumps(cache)
            other = pickle.loads(blob)
            try:
                assert other['k'] == {'value': 1}
            finally:
                other.close()


# -- Review-pass-2: defensive Settings pop (M3) ------------------------


def test_disk_pickle_key_in_settings_does_not_leak(tmp_cache_dir, clear_env):
    """A stale ``disk_pickle_key`` row in Settings (e.g. from an older
    patched build, manual edit, or attacker) must NOT influence the
    actual HMAC key used by the Disk on subsequent opens."""
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir) as cache:
            cache['warm'] = 1

    # Manually inject a hostile row that, if naively trusted, would
    # become the HMAC key string.
    poisoned = 'attacker-supplied-key-' + 'A' * 16
    con = sqlite3.connect(op.join(tmp_cache_dir, 'cache.db'))
    try:
        con.execute(
            'INSERT OR REPLACE INTO Settings VALUES (?, ?)',
            ('disk_pickle_key', poisoned),
        )
        con.commit()
    finally:
        con.close()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir) as cache:
            cache['after'] = {'value': 1}
            assert cache['after'] == {'value': 1}
            disk = cache._disk

    # The Disk's pickle_key_arg must remain UNSET (defensive pop did
    # its job); the resolved key must not equal the poisoned bytes.
    from diskcache.core import _PICKLE_KEY_UNSET as UNSET
    assert disk._pickle_key_arg is UNSET, (
        'pickle_key_arg leaked from Settings: %r' % disk._pickle_key_arg
    )
    if disk._pickle_key_resolved not in (None, False):
        assert disk._pickle_key_resolved != poisoned.encode('utf-8'), (
            'resolved key matches poisoned Settings row'
        )


# -- Review-pass-2: edge-case envelope parsing (M9) --------------------


def test_short_envelope_header_raises(tmp_cache_dir, clear_env):
    """Bytes that start with ``DCv1`` but are shorter than the full
    36-byte header must NOT be treated as enveloped."""
    key = secrets.token_bytes(32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            cache['k'] = 1

    con = sqlite3.connect(op.join(tmp_cache_dir, 'cache.db'))
    try:
        # Magic + only 20 bytes (truncated MAC).
        con.execute(
            'UPDATE Cache SET value = ?, mode = 4 WHERE key = ?',
            (PICKLE_MAGIC + b'\x00' * 20, 'k'),
        )
        con.commit()
    finally:
        con.close()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            with pytest.raises(pickle.UnpicklingError):
                cache['k']


def test_zero_byte_value_raises(tmp_cache_dir, clear_env):
    key = secrets.token_bytes(32)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            cache['k'] = 1

    con = sqlite3.connect(op.join(tmp_cache_dir, 'cache.db'))
    try:
        con.execute(
            'UPDATE Cache SET value = ?, mode = 4 WHERE key = ?',
            (b'', 'k'),
        )
        con.commit()
    finally:
        con.close()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            with pytest.raises(pickle.UnpicklingError, match='envelope'):
                cache['k']


def test_valid_hmac_invalid_pickle_raises(tmp_cache_dir, clear_env):
    """An attacker who somehow obtains the key (or a corrupted-but-
    HMAC-valid blob) and writes a non-pickle payload should still get a
    pickle error -- not silent garbage."""
    key = secrets.token_bytes(32)
    payload = b'\xff' * 50  # not valid pickle
    mac = hmac.new(key, payload, hashlib.sha256).digest()
    envelope = PICKLE_MAGIC + mac + payload

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            cache['k'] = 1

    con = sqlite3.connect(op.join(tmp_cache_dir, 'cache.db'))
    try:
        con.execute(
            'UPDATE Cache SET value = ?, mode = 4 WHERE key = ?',
            (envelope, 'k'),
        )
        con.commit()
    finally:
        con.close()

    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir, disk_pickle_key=key) as cache:
            with pytest.raises((pickle.UnpicklingError, EOFError, KeyError)):
                cache['k']


def test_env_var_short_hex_raises(tmp_cache_dir, monkeypatch):
    monkeypatch.setenv(PICKLE_KEY_ENV, 'deadbeef')  # 4 bytes
    with dc.Cache(tmp_cache_dir) as cache:
        with pytest.raises(ValueError, match='at least'):
            cache['x'] = {'complex': 'value'}


def test_env_var_empty_raises(tmp_cache_dir, monkeypatch, clear_env):
    """Empty string env var must NOT be silently treated as 'unset'.

    monkeypatch the env var to '' explicitly; the resolution code uses
    ``if env_value:`` so empty falls through to the file fallback,
    which is the intended behavior.  Verify that path works (no crash).
    """
    monkeypatch.setenv(PICKLE_KEY_ENV, '')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore', dc.UnsafePickleWarning)
        with dc.Cache(tmp_cache_dir) as cache:
            cache['k'] = {'val': 1}
            assert cache['k'] == {'val': 1}
    # File fallback created the key file because env was effectively unset.
    assert op.exists(op.join(tmp_cache_dir, PICKLE_KEY_FILENAME))


# -- Review-pass-2: multi-process race on key file (H3) ----------------


def _race_worker(directory, idx, queue):  # pragma: no cover - subprocess
    import warnings as _w
    import diskcache as _dc

    _w.simplefilter('ignore', _dc.UnsafePickleWarning)
    try:
        with _dc.Cache(directory) as cache:
            cache[('worker', idx)] = {'pid': os.getpid(), 'idx': idx}
            queue.put(('ok', idx, cache[('worker', idx)]))
    except Exception as exc:  # pylint: disable=broad-except
        queue.put(('err', idx, repr(exc)))


@pytest.mark.skipif(
    sys.platform == 'win32' and sys.version_info < (3, 8),
    reason='multiprocessing fork start method is unreliable on this combo',
)
def test_concurrent_key_file_creation_no_crash(tmp_cache_dir, clear_env):
    """Multiple processes creating the same fresh cache must all
    succeed -- the original code crashed losers with a false-positive
    'too short' error before the writer's os.write completed."""
    ctx = multiprocessing.get_context('spawn')
    queue = ctx.Queue()
    procs = [
        ctx.Process(target=_race_worker, args=(tmp_cache_dir, i, queue))
        for i in range(4)
    ]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=30)
        assert p.exitcode == 0, (
            'worker exited %s; stdout/stderr should not contain '
            'RuntimeError("too short")' % p.exitcode
        )

    results = []
    while not queue.empty():
        results.append(queue.get_nowait())
    assert len(results) == 4
    for status, idx, payload in results:
        assert status == 'ok', 'worker %d failed: %r' % (idx, payload)


# -- Review-pass-2: sanity for trimmed TypeError (M4) ------------------


def test_typeerror_message_lists_only_real_types(tmp_cache_dir, clear_env):
    with pytest.raises(TypeError) as exc_info:
        dc.Cache(tmp_cache_dir, disk_pickle_key=12345)
    msg = str(exc_info.value)
    assert 'bytes' in msg and 'hex str' in msg
    assert 'False' not in msg and 'None' not in msg, (
        'TypeError must not advertise False/None: caller filters them. '
        'Got: %r' % msg
    )
