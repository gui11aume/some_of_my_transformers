#!/usr/bin/env python3

import bz2
import json
import sys

with bz2.open(sys.argv[1]) as f:
   for line in f:
      doc = json.loads(line.decode("ascii"))
      seq = doc["seq"].upper()
      print(" ".join(seq[i:i+6] for i in range(len(seq)-5)))
