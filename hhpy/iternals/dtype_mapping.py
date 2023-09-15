# --- dtypes
dtype_mapping = {
    'Int': ['Int8', 'Int16', 'Int32', 'Int64', 'UInt8', 'UInt16', 'UInt32', 'UInt64'],
    'UInt': ['UInt8', 'UInt16', 'UInt32', 'UInt64'],
    'int': ['int8', 'int16', 'int32', 'int64', 'uint8', 'uint16', 'uint32', 'uint64'],
    'uint': ['uint8', 'uint16', 'uint32', 'uint64'],
    'float': ['float8', 'float16', 'float32', 'float64'],
    'string': ['string'],
    'object': ['object'],
    'boolean': ['boolean'],
    'category': ['category'],
    'datetime': ['datetime64[ns]'],
    'datetimez': ['datetime64[ns, <tz>]'],
    'period': ['period[<freq>]']
}
dtype_mapping['Iint'] = dtype_mapping['Int'] + dtype_mapping['int']
dtype_mapping['number'] = dtype_mapping['Iint'] + dtype_mapping['float']
dtype_mapping['datetime64'] = dtype_mapping['datetime']
