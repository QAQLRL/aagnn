import torch,random,json
import json, bz2, pickle, gzip as gz
import json
import pandas as pd
from pymatgen.entries.computed_entries import ComputedStructureEntry

ratios = [0.2,0.3,0.4,0.5,0.6,0.7,0.8]
data = json.load(open('predicted_mixed_perovskites.json','r')) 
entries = [ComputedStructureEntry.from_dict(i) for i in data]
random.shuffle(entries)

print("Found " + str(len(entries)) + " entries")
e_form=[]
atoms = []
for entry in entries[:1000]:
    e_form.append(entry.data['e-form'])
    atoms.append(entry.structure)
df = pd.DataFrame({'atoms':atoms,'e_form':e_form})
print('save df')
#一直报错
df.to_json('dcgat.json')
print('save pickle')
pickle.dump(entries, gz.open('dcgat_'+'.pickle.gz','wb'))


#only using the first 1000 entries to save time
# for ratio in ratios:
#     size = ratio * len(entries)
#     print(f'length of ratio perovskites is:{size} \n')
#     pickle.dump(entries[:size], gz.open('dcgat_'+str(ratio)+'.pickle.gz','wb'))
#     df[:size].to_json('dcgat_'+str(ratio)+'.json')
