import os
# Travers all the branch of a specified path
# for (root,dirs,files) in os.walk('./logs/GuidedBaselines/out',topdown=True):
# for (root,dirs,files) in os.walk('./logs/GradCamSparseSampling',topdown=True):
for (root,dirs,files) in os.walk('./logs/GradCamSimMaskGT',topdown=True):
  best_epg = 0
  pareto = ''
  params = ''
  for file in files:
    if file.endswith('.out'):
      try:
        with open(os.path.join(root,file),'r') as f:
            first_line = f.readline().strip()
            # print(first_line)
            data = dict(item.strip().split(": ", 1) for item in first_line.split(", "))
            # print(data)
            # fs = first_line.find('SimilarityThreshold: ') + 21
            # if data['Backbone']=='vanilla' and data['Attribution_method']=='GradCam' and data['localization loss']=='Energy' and data['feedback_type']=='bbox':
            if data['Attribution_method']=='GradCam' and data['NumGuidingPoints']=='10.':
              # print(first_line)
              res = f.read()
              s = res.rfind('Pareto Costs:')
              e = res.find('\n',s)
              res_line = res[s:e+1]

              # print(f'Result line: {res_line}')
              results = res_line.strip().split(')')[-2].split(',')[0]
              results_index = results.find('(')
              
              current_epg = float(results[results_index+1:])
              if current_epg > best_epg:
                best_epg = current_epg
                params = first_line
                pareto = res_line
      except Exception as e:
        print(f'Error reading file {file}: {e}')
            
print(f'Best EPG: {best_epg}')
print(f'Params: {params}')
print(f'Pareto: {pareto.replace(")","),")[:-2]}')