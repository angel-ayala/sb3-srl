from setuptools import setup

package_name = 'sb3_srl'

setup(
    name=package_name,
    version='1.0.0',    
    description="Code extension of Stable Baselines 3 Reinforcement Learning algorithms with State Representation Learning methods",
    url='https://github.com/angel-ayala/sb3-srl',
    author='Angel Ayala',
    author_email='aaam@ecomp.poli.br',
    license='GPL-3.0',
    packages=[package_name],
    install_requires=[
         'gymnasium==0.29.1',
         'stable_baselines3>=2.6.0'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Education',
        'License :: OSI Approved :: GNU General Public License v3 (GPLv3)',  
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.10',
        'Topic :: Games/Entertainment :: Simulation',
    ],
)
