#
# Copyright (c) 2020 jintian.
#
# This file is part of CenterNet_Pro_Max
# (see jinfagang.github.io).
#
# Licensed to the Apache Software Foundation (ASF) under one
# or more contributor license agreements.  See the NOTICE file
# distributed with this work for additional information
# regarding copyright ownership.  The ASF licenses this file
# to you under the Apache License, Version 2.0 (the
# "License"); you may not use this file except in compliance
# with the License.  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an
# "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
# KIND, either express or implied.  See the License for the
# specific language governing permissions and limitations
# under the License.
#
import tabulate
import itertools



def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


def create_table_with_header(header_dict, headers=["category", "AP"], min_cols=6):
    """
    create a table with given header.

    Args:
        header_dict (dict):
        headers (list):
        min_cols (int):

    Returns:
        str: the table as a string
    """
    assert min_cols % len(headers) == 0, "bad table format"
    num_cols = min(min_cols, len(header_dict) * len(headers))
    result_pair = [x for pair in header_dict.items() for x in pair]
    row_pair = itertools.zip_longest(*[result_pair[i::num_cols] for i in range(num_cols)])
    table = tabulate(
        row_pair, tablefmt="pipe", floatfmt=".3f",
        headers=headers * (num_cols // len(headers)),
        numalign="left")
    return table
