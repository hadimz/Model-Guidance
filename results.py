import os

# Travers all the branch of a specified path
for (root,dirs,files) in os.walk('./logs/GuidedBaselines/out',topdown=True):
  best_f1 = 0
  pareto = ''
  params = ''
  for file in files:
    if file.endswith('.out'):
        with open(os.path.join(root,file),'r') as f:
          first_line = f.readline().strip()
          data = dict(item.strip().split(": ", 1) for item in first_line.split(", "))
          # print(data)
          # fs = first_line.find('SimilarityThreshold: ') + 21
          if data['Backbone']=='bcos' and data['Attribution_method']=='GradCam' and data['localization loss']=='Energy' and data['feedback_type']=='bbox':
            # print(first_line)
            res = f.read()
            s = res.rfind('Pareto Costs:')
            e = res.find('\n',s)
            res_line = res[s:e+1]

            results = res_line.strip().split(')')[-2].split(',')[0]
            results_index = results.find('(')
            current_f1 = float(results[results_index+1:])
            if current_f1 > best_f1:
              best_f1 = current_f1
              params = first_line
              pareto = res_line
    
            
print(f'Best F1: {best_f1}')
print(f'Params: {params}')
print(f'Pareto: {pareto.replace(")","),")[:-2]}')
          
