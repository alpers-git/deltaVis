import subprocess

build_step = 'cmake --build ./build'

commands = [
    '/home/alpers/Desktop/Dev/deltaVis/build/deltaVisViewer ../../Data/jets.umesh -tf ./transferfunction.tf -cam 2620.044922 6299.418945 3403.929932 2619.529053 6298.740234 3403.407227 -0.457003 0.734125 -0.502204 0.724668',
    '/home/alpers/Desktop/Dev/deltaVis/build/deltaVisViewer ../../Data/jets.umesh -cam 74.545273 72.365623 283.063324 74.457329 72.332382 282.067749 0.013214 0.999316 -0.034533 0.724668 -tf ./transferfunction.tf -shadows 0.003 -0.9 -0.7'
]

# run the first command
print('Building...')
subprocess.call(build_step, shell=True)

mc = 8

numRays = [4, 8, 12, 16]
shadowArg = '-shadows 0.0 -1.0 0.5'

print('Running...')
for ray in numRays:
    for i in range(0, 2):
        command = commands[0] + ' -numRays ' + str(ray)
        if i == 1:
            print(f'Running with numRays {ray} and shadows on...')
            command = command + ' -numRays ' + str(ray) + ' ' + shadowArg
        else:
            print(f'Running with numRays {ray} and shadows off...')
        subprocess.call(command, shell=True)