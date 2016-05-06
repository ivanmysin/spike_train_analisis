# -*- coding: utf-8 -*-
def time2sec(time):
    vals = time.split(":")
    sec = 60*60*float(vals[0]) + 60*float(vals[1]) + float(vals[2])
    return sec
def get_bounds(csvfilepath):
    f = open(csvfilepath, "r")
    csvcontent = f.read()
    csvlines = csvcontent.split("\n")
    
    bounds = []
    
    for line in csvlines[1:]:
        if (line == ""):
            continue
        
        vals = line.split(",")
        if (len(vals) < 5):
            continue
        bound = {
            "channel" : int(vals[1]),
            "low_bound" : time2sec(vals[2]),
            "upper_bound" : time2sec(vals[3]),
            "comment" : vals[4],
        }
        bounds.append(bound)
    
    f.close()
    return bounds
