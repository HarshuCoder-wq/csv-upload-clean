import mysql.connector
db_hosts = {
    'prod': {
        'host': 'geetanshmehra.com',
        'user': 'geetanshmehra_usr',
        'password': '*Dsl.1uGk.w-',
        'database': 'geetanshmehra_db'
    },
    'backup': {
        'host': 'fasttracksms.com',
        'user': 'i9710084_ukiq1',
        'password': 'D.AlCsACtnyb9yNFW9P50',
        'database': 'i9710084_ukiq1'
    }
}

def get_db_connection(host_key='prod'):
    config = db_hosts.get(host_key)
    if not config:
        raise Exception(f"Unknown DB host: {host_key}")
    return mysql.connector.connect(**config)