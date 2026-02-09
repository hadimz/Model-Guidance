import os
# Travers all the branch of a specified path
# for (root,dirs,files) in os.walk('./logs/GuidedBaselines/out',topdown=True):
# for (root,dirs,files) in os.walk('./logs/GradCamSparseSampling',topdown=True):
# for (root,dirs,files) in os.walk('./logs/GradCamSimMaskGT',topdown=True):
best_epg = 0
pareto_epg = ''
params_epg = ''
val_metrics_epg = ''

best_f1 = 0
pareto_f1 = ''
params_f1 = ''
val_metrics_f1 = ''
for (root,dirs,files) in os.walk('./logs/COCO_Final/points_adaptive/vanilla',topdown=True):
# for (root,dirs,files) in os.walk('./logs/COCO_Final/points/vanilla',topdown=True):
# for (root,dirs,files) in os.walk('./logs/COCO_Final/GuidedBaselines/vanilla',topdown=True):  
  for file in files:
    if file.endswith('.out'):
      try:
        with open(os.path.join(root,file),'r') as f:
            first_line = f.readline().strip()
            if 'feedback_type_points_adaptive_' in file and 'SimilarityThreshold_0.99_' in file: # and 'SimilarityThreshold_0.99_' in file:  and 'numGuidingPoints_25_' in file
            # if 'feedback_type_points' in file and 'SimilarityThreshold_0.99_' in file: # and 'SimilarityThreshold_0.99_' in file:  and 'numGuidingPoints_25_' in file
            # if "feedback_type_mask" in file and 'localization_loss_Energy' in file:
              res = f.read()
              s = res.rfind('Pareto Costs:')
              e = res.find('\n',s)
              res_line = res[s:e+1]

              # print(f'Result line: {res_line}')
              results = res_line.strip().split(')')[-2].split(',')[1]
              results_index = results.find('(')
              current_epg = float(results[results_index+1:])
              if current_epg > best_epg:
                best_epg = current_epg
                params_epg = file
                pareto_epg = res_line
                
                val_metrics_epg = res[res.find('Loading test dataset from datasets/COCO2014/processed'):]
                val_metrics_epg = res[res.find('average guiding points per image:'):]
              
              results = res_line.strip().split(')')[-2].split(',')[0]
              results_index = results.find('(')
              current_f1 = float(results[results_index+1:])
              if current_f1 > best_f1:
                best_f1 = current_f1
                params_f1 = file
                pareto_f1 = res_line
                val_metrics_f1 = res[res.find('Loading test dataset from datasets/COCO2014/processed'):]
                val_metrics_f1 = res[res.find('average guiding points per image:'):]
                
      except Exception as e:
        print(f'Error reading file {file}: {e}')

print(f'Best F1: {best_f1}')
print(f'Params: {params_f1}')
print(f'Pareto: {pareto_f1.replace(")","),")[:-2]}')
print('Validation Metrics:', val_metrics_f1)

print(f'Best EPG: {best_epg}')
print(f'Params: {params_epg}')
print(f'Pareto: {pareto_epg.replace(")","),")[:-2]}')
print('Validation Metrics:', val_metrics_epg)