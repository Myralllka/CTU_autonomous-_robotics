from __future__ import absolute_import, division, print_function
import json
import os
import pickle


__all__ = [
    'makedirs',
    'read_binary',
    'read_json',
    'read_object',
    'read_text',
    'write_binary',
    'write_json',
    'write_object',
    'write_text'
]


def makedirs(path):
    try:
        os.makedirs(path, exist_ok=True)
    except OSError:
        pass


def read_binary(path):
    assert isinstance(path, str)
    with open(path, 'rb') as f:
        return f.read()


def write_binary(path, s):
    assert isinstance(path, str)
    assert isinstance(s, (bytes, str))
    makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        f.write(s)


def read_text(path, encoding='utf-8'):
    assert isinstance(path, str)
    assert isinstance(encoding, str)
    return read_binary(path).decode(encoding)


def write_text(path, s, encoding='utf-8'):
    assert isinstance(path, str)
    if not isinstance(s, str):
        print(type(s))
    assert isinstance(s, str)
    assert isinstance(encoding, str)
    makedirs(os.path.dirname(path))
    write_binary(path, s.encode(encoding))


def read_object(path):
    assert isinstance(path, str)
    with open(path, 'rb') as f:
        return pickle.load(f)


def write_object(path, obj):
    assert isinstance(path, str)
    makedirs(os.path.dirname(path))
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def read_json(path):
    assert isinstance(path, str)
    with open(path, 'r') as fp:
        return json.load(fp)


def write_json(path, obj):
    assert isinstance(path, str)
    makedirs(os.path.dirname(path))
    with open(path, 'w') as fp:
        json.dump(obj, fp, indent=1, sort_keys=True)
