import random

from pymilvus import (
    MilvusClient,
    FieldSchema, CollectionSchema, DataType,
    Collection,
    utility
)

# This example shows how to:
#   1. connect to Milvus server
#   2. create a collection
#   3. insert entities
#   4. create index
#   5. search

_HOST = '127.0.0.1'
_PORT = '19530'
_URI = f"http://{_HOST}:{_PORT}"

# Const names
_COLLECTION_NAME = 'demo'
_ID_FIELD_NAME = 'id_field'
_VECTOR_FIELD_NAME = 'float_vector_field'

# Vector parameters
_DIM = 128
_INDEX_FILE_SIZE = 32  # max file size of stored index

# Index parameters
_METRIC_TYPE = 'L2'
_INDEX_TYPE = 'IVF_FLAT'
_NLIST = 1024
_NPROBE = 16
_TOPK = 3

milvus_client: MilvusClient = None
# Create a Milvus connection
def create_connection():
    global milvus_client
    print(f"\nCreate connection...")
    milvus_client = MilvusClient(uri=_URI,
                            secure=True,
                            server_pem_path='/root/ibm-lh-dev/localstorage/volumes/infra/tls/cert.crt',
                            server_name='localhost',)
    # milvus_client = MilvusClient(host=_HOST, port=_PORT, secure=True, server_pem_path="/root/milvus/milvus/configs/cert/server.pem", server_name="localhost")
    print(f"\nList connections:")
    print(milvus_client._get_connection())


# Create a collection named 'demo'
def create_collection(name, id_field, vector_field):
    field1 = FieldSchema(name=id_field, dtype=DataType.INT64, description="int64", is_primary=True)
    field2 = FieldSchema(name=vector_field, dtype=DataType.FLOAT_VECTOR, description="float vector", dim=_DIM,
                         is_primary=False)
    schema = CollectionSchema(fields=[field1, field2], description="collection description")
    milvus_client.create_collection(collection_name=name,schema=schema)
    milvus_client.describe_collection(collection_name=name)
    # collection = Collection(name=name, data=None, schema=schema)
    print("\ncollection created:", name)
    # return collection


def has_collection(name):
    # global milvus_client
    return milvus_client.has_collection(name)


# Drop a collection in Milvus
def drop_collection(name):
    milvus_client.drop_collection(name)
    # collection = Collection(name)
    # collection.drop()
    print("\nDrop collection: {}".format(name))


# List all collections in Milvus
def list_collections():
    print("\nlist collections:")
    print(milvus_client.list_collections())


def insert(name,num, dim):
    data_dict = []
    for i in range(num):
        entity = {
        "id_field": i+1,  # Assuming id_field is the name of the field corresponding to the ID
        "float_vector_field": [random.random() for _ in range(dim)]
    }
        data_dict.append(entity)
    # print("\n",data_dict)
    insert_result = milvus_client.insert(collection_name=name,data=data_dict)
    # return data[1]
    return insert_result,data_dict[1]


def get_entity_num(insert_result):
    print(f"\nThe number of entity: {insert_result['insert_count']}")


def create_index(filed_name,name):
    index_params = milvus_client.prepare_index_params()

    index_params.add_index(
        field_name=filed_name, 
        index_type=_INDEX_TYPE,
        metric_type=_METRIC_TYPE,
        params={"nlist": _NLIST}
    )

    milvus_client.create_index(
        collection_name=name,
        index_params=index_params
    )
    print("\nCreated index\n")


def drop_index(name,field_name):
    milvus_client.drop_index(name,index_name=field_name)
    print("\nDrop index sucessfully")


def load_collection(name):
    milvus_client.load_collection(name)


def release_collection(name):
    milvus_client.release_collection(name)


def search(name, vector_field, id_field, search_vectors):
    search_param = {
        # "data": search_vectors,
        "anns_field": vector_field,
        "param": {"metric_type": _METRIC_TYPE, "params": {"nprobe": _NPROBE}},
        "expr": f"{id_field} > 0"}
    # results = collection.search(**search_param)
    results = milvus_client.search(collection_name=name,data=search_vectors,limit= _TOPK,search_params=search_param)
    for i, result in enumerate(results):
        print("\nSearch result for {}th vector: ".format(i))
        for j, res in enumerate(result):
            print("Top {}: {}".format(j, res))


def main():
    # create a connection
    create_connection()

    has = milvus_client.has_collection("hello_milvus")
    print(f"Does collection hello_milvus exist in Milvus: {has}")
    # drop collection if the collection exists
    if has_collection(_COLLECTION_NAME):
        drop_collection(_COLLECTION_NAME)

    # create collection
    # collection = 
    create_collection(_COLLECTION_NAME, _ID_FIELD_NAME, _VECTOR_FIELD_NAME)

    # show collections
    list_collections()

    # # insert 10000 vectors with 128 dimension
    insert_result,vector = insert(_COLLECTION_NAME,10000, _DIM)

    # # get the number of entities
    get_entity_num(insert_result)

    # create index
    create_index( _VECTOR_FIELD_NAME,_COLLECTION_NAME)

    # load data to memory
    load_collection(_COLLECTION_NAME)
    vectors = [vector["float_vector_field"]]
    # search
    search(_COLLECTION_NAME,_VECTOR_FIELD_NAME, _ID_FIELD_NAME, vectors)

    # release memory
    release_collection(_COLLECTION_NAME)

    # drop collection index
    drop_index(_COLLECTION_NAME, _VECTOR_FIELD_NAME)

    # drop collection
    drop_collection(_COLLECTION_NAME)


if __name__ == '__main__':
    main()