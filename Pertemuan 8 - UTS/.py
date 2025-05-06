import numpy as np
import os
import traceback
from pyspark.sql import SparkSession

# Set environment variables agar worker dan driver pakai versi Python yang sama
os.environ["PYSPARK_PYTHON"] = "C:\\Python312\\python.exe"
os.environ["PYSPARK_DRIVER_PYTHON"] = "C:\\Python312\\python.exe"

# Inisialisasi SparkSession dengan 5 worker lokal
spark = SparkSession.builder \
    .appName("MatrixMultiply") \
    .master("local[5]") \
    .getOrCreate()

# Atur log level supaya bisa lihat info debug saat crash
spark.sparkContext.setLogLevel("DEBUG")

def matrix_multiply_spark(matrix_a, matrix_b):
    if matrix_a.shape[1] != matrix_b.shape[0]:
        raise ValueError("Dimensi matriks tidak sesuai untuk perkalian")

    sc = spark.sparkContext

    # Ubah numpy array ke RDD dan bagi jadi partisi lebih kecil
    rdd_a = sc.parallelize(matrix_a.tolist(), numSlices=20).zipWithIndex()
    rdd_b = sc.parallelize(matrix_b.T.tolist(), numSlices=20).zipWithIndex()  # transpose supaya bisa dijumlahkan baris ke kolom

    # Cartesian product dan perkalian baris x kolom
    result = rdd_a.cartesian(rdd_b) \
        .map(lambda x: ((x[0][1], x[1][1]), sum(a*b for a, b in zip(x[0][0], x[1][0])))) \
        .reduceByKey(lambda a, b: a + b) \
        .map(lambda x: (x[0][0], (x[0][1], x[1]))) \
        .groupByKey() \
        .map(lambda x: (x[0], sorted(list(x[1]), key=lambda y: y[0]))) \
        .sortByKey() \
        .map(lambda x: [v for (_, v) in x[1]])

    return np.array(result.collect())

# Matriks 1000x1000
matrix_a = np.random.rand(1000, 1000)
matrix_b = np.random.rand(1000, 1000)

# Jalankan fungsi dalam try-except agar lebih aman
try:
    result = matrix_multiply_spark(matrix_a, matrix_b)
    print("Hasil 5x5 pertama:")
    print(result[:5, :5])
except Exception as e:
    print("Terjadi error saat menjalankan Spark:")
    traceback.print_exc()