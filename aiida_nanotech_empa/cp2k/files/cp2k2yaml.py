import re
import numpy as np
import yaml

def CP2K2DICT(input_lines=[]):
    input_dict={}
    lines=input_lines #.splitlines()
    i_am_at=[input_dict]
    for l in lines:
        fc=len(l) - len(l.lstrip())
        if l[fc] not in ['#','!']:
            has_arg=len(l.split())>1
            key=l.split()[0].lstrip('&')
            if "&" in l and 'END' not in l:
                if key in i_am_at[-1].keys():
                    if type(i_am_at[-1][key]) is not type([]):
                        i_am_at[-1][key]=[i_am_at[-1][key]]
                    if has_arg:
                        i_am_at[-1][key].append({'_':" ".join([str(s) for s in l.split()[1:] ])})
                    else:
                        i_am_at[-1][key].append({})
                    i_am_at.append(i_am_at[-1][key][-1])
                else:
                    if has_arg:
                        i_am_at[-1][key]={'_':" ".join([str(s) for s in l.split()[1:] ])}
                    else:
                        i_am_at[-1][key]={}
                    i_am_at.append(i_am_at[-1][key])                
            elif "&" in l and 'END'  in l:
                if type(i_am_at[-1]) is not type([]):
                    i_am_at = i_am_at[0:-1]
                else:
                    i_am_at = i_am_at[0:-len(i_am_at[-1])]
            else:
                while key in i_am_at[-1].keys(): #fixes teh case where inside &KINDS we have BASIS .. and BASIS RI_AUX ..
                    key = key+' '
                if has_arg:
                    i_am_at[-1].update({key:" ".join([str(s) for s in l.split()[1:] ])})
                else:
                    i_am_at[-1].update({key:''})
    if len(i_am_at) > 1:
        msg='there are unclosed cards'
    else:
        msg=''
       
    return msg,input_dict 

with open('cp2k.inp') as f:
    all_lines = f.readlines()
yaml.SafeDumper.org_represent_str = yaml.SafeDumper.represent_str
def repr_str(dumper, data):
    if '\n' in data:
        # To make this work, we can't have whitespace before newline
        data = ''.join([line.rstrip()+'\n' for line in data.splitlines()])
        return dumper.represent_scalar(u'tag:yaml.org,2002:str', data, style='|')
    return dumper.org_represent_str(data)
yaml.add_representer(str, repr_str, Dumper=yaml.SafeDumper)
# -------------
msg,the_dict = CP2K2DICT(input_lines=all_lines)
with open('cp2k.yml', 'w', encoding='utf8') as yaml_file:
    yaml.safe_dump(the_dict, yaml_file, default_flow_style=False, encoding='utf8', allow_unicode=True, sort_keys=False)