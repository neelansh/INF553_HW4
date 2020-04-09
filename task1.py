import os
import sys
from pyspark.sql import SQLContext
from pyspark import SparkContext, SparkConf
from graphframes import  *
import json
import itertools
import math
import time


appName = 'assignment4'
master = 'local[*]'
conf = SparkConf().setAppName(appName).setMaster(master).set('spark.jars.packages','graphframes:graphframes:0.6.0-spark2.3-s_2.11')
sc = SparkContext(conf=conf)
sc.setLogLevel("ERROR")
sqlContext = SQLContext(sc)

def save_output(output_path, output):
    output = sorted([x[1] for x in output], key=lambda x: x[0])
    output = sorted(output, key=lambda x: len(x))
    output = ["'"+"', '".join(x)+"'\n" for x in output]
    file = open(output_path, 'wt')
    for line in output:
        file.write(line)
        
    file.close()
    return 

if __name__ == '__main__':
    input_path = sys.argv[1].strip()
    edges = sc.textFile(input_path).map(lambda x: tuple(x.split())).collect()
    edges_reversed = sc.textFile(input_path).map(lambda x: tuple(x.split())).map(lambda x: (x[1], x[0])).collect()
    vertices = sc.textFile(input_path).flatMap(lambda x: x.split()).distinct().collect()
    vertices = sqlContext.createDataFrame([(x,) for x in vertices], ['id'])
    edges = sqlContext.createDataFrame(edges+edges_reversed, ['src', 'dst'])

    graph = GraphFrame(vertices, edges)

    result = graph.labelPropagation(maxIter=5)
    result = result.rdd.map(lambda x: (x['label'], x['id'])).groupByKey().map(lambda x: (x[0], sorted(list(x[1]))))
#     length_communities = result.count()
#     result_sorted = result.takeOrdered(length_communities, lambda x: len(x[1]))
    result_sorted = result.collect()

    save_output(sys.argv[2].strip(), result_sorted)