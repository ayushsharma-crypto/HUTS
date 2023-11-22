from scipy.io import loadmat
from collections import namedtuple
from sys import argv

dbStruct = namedtuple('dbStruct', ['whichSet', 'dataset',
    'dbImage', 'locDb', 'qImage', 'locQ', 'numDb', 'numQ',
    'posDistThr', 'posDistSqThr'])

def parse_dbStruct(path):
    mat = loadmat(path)

    matStruct = mat['dbStruct'][0]

    dataset = 'dataset'

    whichSet = 'VPR'

    dbImage = matStruct[0]
    locDb = matStruct[1]

    qImage = matStruct[2]
    locQ = matStruct[3]

    numDb = matStruct[4].item()
    numQ = matStruct[5].item()

    posDistThr = matStruct[6].item()
    posDistSqThr = matStruct[7].item()

    return dbStruct(whichSet, dataset, dbImage, locDb, qImage, 
            locQ, numDb, numQ, posDistThr, 
            posDistSqThr)

print(parse_dbStruct(argv[1]))