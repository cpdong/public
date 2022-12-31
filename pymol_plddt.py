#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 01 2023
@author: cpdong
"""

from pymol import cmd

def plddt(selection="all"):
    cmd.color("#0053D6", f"({selection}) and b > 90")
    cmd.color("#65CBF3", f"({selection}) and b < 90 and b > 70")
    cmd.color("#FFDB13", f"({selection}) and b < 70 and b > 50")
    cmd.color("#FF7D45", f"({selection}) and b < 50")
    
cmd.extend("plddt", plddt)
cmd.auto_arg[0]["plddt"] = [cmd.object_sc, "object", ""]
