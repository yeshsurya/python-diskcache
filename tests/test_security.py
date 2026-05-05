"""Tests for the CVE-2025-69872 / GHSA-w8v5-vhqr-4h9v mitigation.

These tests exercise the HMAC envelope wrapping every pickle blob written
by :class:`diskcache.Disk`.  They live in their own module so the rest of
the suite continues to run with the default
:class:`UnsafePickleWarning` filter, while these tests can opt to either
suppress or assert that the warning is emitted.
"""

import io
import os
import os.path as op
import pickle
import secrets
import shutil
import sqlite3
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
