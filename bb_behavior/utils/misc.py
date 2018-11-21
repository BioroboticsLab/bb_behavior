def generate_64bit_id():
    """Returns a unique ID that is 64 bits long.

    Taken from the bb_pipeline codebase.
    """
    import hashlib, uuid

    hasher = hashlib.sha1()
    hasher.update(uuid.uuid4().bytes)
    hash = int.from_bytes(hasher.digest(), byteorder='big')
    # strip to 64 bits
    hash = hash >> (hash.bit_length() - 64)
    return hash