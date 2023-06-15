#!/bin/sh
save_dir="/nfs/turbo/coe-jag/zjsun/data/nextiajd_datasets"
curl https://mydisk.cs.upc.edu/s/eCmfrNEBSKkcWcn/download -o ${save_dir}/testbedXS.zip
curl https://mydisk.cs.upc.edu/s/dX3FajwWZn7rrrd/download -o ${save_dir}/testbedS.zip
curl https://mydisk.cs.upc.edu/s/niPyR4WTtxydprj/download -o ${save_dir}/testbedM.zip
curl https://mydisk.cs.upc.edu/s/4qoi76ziT2wJaCR/download -o ${save_dir}/testbedL.zip
unzip ${save_dir}/testbedXS.zip -d ${save_dir}
unzip ${save_dir}/testbedS.zip -d ${save_dir}
unzip ${save_dir}/testbedM.zip -d ${save_dir}
unzip ${save_dir}/testbedL.zip -d ${save_dir}
rm -r ${save_dir}/__MACOSX
