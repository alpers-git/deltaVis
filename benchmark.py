import subprocess

build_step = 'cmake --build ./build'

# UPDATE ME
dataset_name = 'earthquake' #'impact6k' # #'scivis2011' 
dataset_num = 2 # 0-indexed
mc_count = 16

commands = [
    '/home/alpers/Desktop/Dev/deltaVis/build/deltaVisViewer ../../Data/impact-6k.umesh -tf ./impact.tf -cam 2620.044922 6299.418945 3403.929932 2619.529053 6298.740234 3403.407227 -0.457003 0.734125 -0.502204 0.724668',
    '/home/alpers/Desktop/Dev/deltaVis/build/deltaVisViewer ../../Data/scivis2011.umesh -tf ./scivis.tf -cam -0.002447 -0.543520 0.464145 0.007299 0.214288 -0.188265 -0.011320 0.652481 0.757721 0.707107',
    '/home/alpers/Desktop/Dev/deltaVis/build/deltaVisViewer ../../Data/earthquake.umesh -tf ./earthquake.tf -cam -1.214427 -0.370428 2.658465 -0.797263 -0.236263 1.750274 -0.014261 0.999416 0.031046 0.707107',
]

print('Building...')
subprocess.call(build_step, shell=True)

numRays = [4, 8, 12, 16]
shadowArg = [
    '-shadows 0.0 -1.0 0.5',
    '-shadows -0.35 -0.2 -0.11',
    '-shadows 0.57 -0.91 0.25',
]

print('Running...')
for i in range(0, 2):
    for ray in numRays:
        command = commands[dataset_num] + ' -numRays ' + str(ray)
        if i == 0:
            print(f'Running with numRays {ray} and shadows on...')
            command = command + ' -numRays ' + str(ray) + ' ' + shadowArg[dataset_num]
        else:
            print(f'Running with numRays {ray} and shadows off...')
        
        # CHANGE MY MACROCELL NUMBER
        log_filename = f'{dataset_name}_{mc_count}x{mc_count}x{mc_count}_' + ('no' if i == 0 else '') + 'shadows' + f'_numRays{ray}.txt'
        subprocess.call(command + f' > ./benchmark_logs/{log_filename}', shell=True)