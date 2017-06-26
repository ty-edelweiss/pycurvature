#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import psycopg2
import pandas as pd
import pandas.io.sql as psql

from pycurvature.dataframe.curvature import Curvature

if __name__ == "__main__":
    with open("./test.json", "r") as f:
        test = json.load(f)

    connection = psycopg2.connect(database=test["database"], host="127.0.0.1", port=5432, user=test["user"], password=test["password"])

    test_table = test["table"]
    limitter = True
    sql = f"SELECT * FROM {test_table}"
    sql = sql + " LIMIT 100" if limitter else sql
    df = psql.read_sql(sql, connection)

    x, y = "longitude", "latitude"
    curvature = Curvature(npo=1).fit(df, [x, y])
    pdf = curvature.predict(df, [-1, 0, 1])

    print(curvature.curvs_)

    print(pdf[[x, y, "labels"]])

    print(pdf["labels"].head())
