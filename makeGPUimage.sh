#!/bin/bash

# at moment can't use g2 from starcluster, so...
# search AMIs in region of interest for ubuntu/images/hvm/ubuntu-precise-12.04
# choose latest and try to launch; may need to work backwards as some AMIs don't support g2.2xlarge
# oregon as of feb 2014: ami-d4d8b8e4, but doesn't work
# ami-52b22962 (20131114) did work
# launch from console
# look in "connect" button to find the ssh command: it will be like the following:
# ssh -i ~/.ssh/ec2star.rsa ec2-user@ec2-54-203-81-145.us-west-2.compute.amazonaws.com
# ecstar.rsa for oregon region, ecstar.rsa-east for east region
export ip=54-184-69-23

ssh -i ~/.ssh/ec2star.rsa ubuntu@ec2-${ip}.us-west-2.compute.amazonaws.com

sudo su

# CRAN repo
apt-key adv --keyserver keyserver.ubuntu.com --recv-keys E084DAB9 
echo "deb http://cran.cnr.berkeley.edu/bin/linux/ubuntu precise/" > \
        /etc/apt/sources.list.d/cran.list 

wget http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1204/x86_64/cuda-repo-ubuntu1204_5.5-0_amd64.deb
dpkg -i cuda-repo-ubuntu1204_5.5-0_amd64.deb

apt-get update
apt-get -y upgrade
# keep grub as is

apt-get install -y nfs-kernel-server nfs-common
ln -s /etc/init.d/nfs-kernel-server /etc/init.d/nfs


# apt-get install r-cran-rmpi # for Rmpi & openMPI - try this?
apt-get install -y emacs git r-recommended libopenmpi-dev libeigen3-dev curl curlftpfs libcurl4-openssl-dev openmpi-bin libopenblas-base libopenblas-dev octave3.2 ipython python-numpy python-scipy python-pandas python-matplotlib r-mathlib sqlite3

# cp libopenblas-base_0.1alpha2.2-3.2_amd64.deb libopenblas-dev_0.1alpha2.2-3.2_amd64.deb from scf to cloud machine:
#rsync -av paciorek@gandalf.berkeley.edu:/server/install/linux/PACKAGES-12.04/libopenblas*2.2-3.2*deb /tmp/
#cd /tmp
#PKGS="libopenblas-base_0.1alpha2.2-3.2_amd64.deb libopenblas-dev_0.1alpha2.2-3.2_amd64.deb"
#dpkg -i ${PKGS}

update-alternatives --set liblapack.so.3gf /usr/lib/lapack/liblapack.so.3gf
# need this or R tries to load an ATLAS function because liblapack.so.3gf points to atlas lapack (which is not installed)

R --no-save <<EOF 
pkgs <- c("Rcpp", "Matrix", "inline", "ggplot2", "plyr", "knitr", "lme4", "devtools", "DBI", "RSQLite", "foreach", "Rmpi", "doMPI", "doParallel", "iterators", "rlecuyer", "reshape2", "glmnet", "pbdDEMO", "pbdSLAP", "pbdMPI", "pbdBASE", "pbdPROF", "pbdDMAT", "RcppArmadillo", "RcppEigen", "bitops")
install.packages(pkgs, repos = "http://cran.cnr.berkeley.edu")
EOF
# failed with lock on Rmpi and glmnet but simply trying again with Rmpi,glmnet,doMPI was fine
# on a separate try it worked fine

apt-get install -y cuda-5-5  
# takes a while

# it suggests to reboot, so reboot VM via EC2 console
exit
ssh -i  ....

sudo su


# this creates /usr/local/cuda-5.5
# it has lib64, no lib

# happens automatically?
ln -s /usr/local/cuda-5.5 /usr/local/cuda

echo "" >> ~root/.bashrc
echo "export PATH=${PATH}:/usr/local/cuda/bin" >> ~root/.bashrc
echo "" >> ~ubuntu/.bashrc
echo "export PATH=${PATH}:/usr/local/cuda/bin" >> ~ubuntu/.bashrc
echo "" >> ~root/.bashrc
echo "alias gtop=\"nvidia-smi -q -g 0 -d UTILIZATION -l 1\"" >> ~root/.bashrc
echo "" >> ~ubuntu/.bashrc
echo "alias gtop=\"nvidia-smi -q -g 0 -d UTILIZATION -l 1\"" >> ~ubuntu/.bashrc
echo "" >> ~ubuntu/.bashrc

# create deviceQuery executable
nvcc deviceQuery.cpp -I/usr/local/cuda/include -I/usr/local/cuda-5.5/samples/common/inc -o /usr/local/cuda/bin/deviceQuery


source ~/.bashrc


echo "/usr/local/cuda/lib64" >> /etc/ld.so.conf.d/cuda.conf
ldconfig

nvidia-smi -q
gtop
# this checks we can access the gpu

cd /usr/src
mkdir magma
cd magma
wget http://icl.cs.utk.edu/projectsfiles/magma/downloads/magma-1.4.1.tar.gz
tar -xvzf magma-1.4.1.tar.gz
cd magma-1.4.1
# note I added -fPIC per the magma README to enable creation of a shared object
scp paciorek@smeagol.berkeley.edu:~/staff/projects/gpus/make.inc.ubuntu.openblas.kepler make.inc

make 2>&1 | tee make.log

make shared 2>&1 | tee make.shared.log

# good to test dgemm: 
# ./testing/testing_dgemm

mkdir /usr/local/magma
make install prefix=/usr/local/magma

# also need magma-1.3.0 for R's magma pkg
cd /usr/src
mkdir magma-1.3.0
cd magma-1.3.0
wget http://icl.cs.utk.edu/projectsfiles/magma/pubs/magma-1.3.0.tar.gz
tar -xvzf magma-1.3.0.tar.gz
cd magma-1.3.0
# for kepler only: per bug report at http://icl.cs.utk.edu/magma/forum/viewtopic.php?f=2&t=900
sed 's/-DGPUSHMEM=300 -arch sm_35/-DGPUSHMEM=300 -gencode arch=compute_30,code=sm_30 -gencode arch=compute_35,code=sm_35/g' Makefile.internal > /tmp/tmp
mv /tmp/tmp Makefile.internal
# note I added -fPIC per the magma README to enable creation of a shared object
scp paciorek@smeagol.berkeley.edu:~/staff/projects/gpus/make.inc.ubuntu.openblas.1.3.0.kepler make.inc
# for 1.3.0 I needed to add -DCUBLAS_GFORTRAN (like in mkl-gcc make.inc)

make 2>&1 | tee make.log

sed 's/\$(MAGMA_DIR)\/lib\/pkgconfig\/magma.pc/\$(MAGMA_DIR)\/lib\/pkgconfig\/magma.pc.in/g' Makefile > /tmp/tmp
mv /tmp/tmp Makefile

make install prefix=/usr/local/magma-1.3.0

# good to test dgemm: 
# ./testing/testing_dgemm

echo "/usr/local/magma/lib" >> /etc/ld.so.conf.d/magma.conf
ldconfig
# I don't think this will cause an issue with the R magma .so as R magma uses .a from magma 1.3.0 so no run-time linking for magma


cd /tmp
wget http://cran.r-project.org/src/contrib/magma_1.3.0-2.tar.gz
R CMD INSTALL  --configure-args="
       --with-cuda-home=/usr/local/cuda \
       --with-magma-lib=/usr/local/magma-1.3.0/lib" \
     magma_1.3.0-2.tar.gz 2>&1 | tee Rmagma.log



cd /usr/src

git clone https://github.com/duncantl/RCUDA
git clone https://github.com/omegahat/RAutoGenRunTime

cd RCUDA/src
ln -s ../../RAutoGenRunTime/src/RConverters.c .
ln -s ../../RAutoGenRunTime/inst/include/RConverters.h .
ln -s ../../RAutoGenRunTime/inst/include/RError.h .

cd ../..

R CMD build RCUDA
R CMD build RAutoGenRunTime
R CMD INSTALL RAutoGenRunTime_0.3-0.tar.gz 
R CMD INSTALL RCUDA_0.4-0.tar.gz 

apt-get install -y python-pip

pip install pycuda
# ignore warning msg

echo "  This is an Ubuntu 12.04 (Precise) based image with GPU support" >> /etc/motd.tail
echo "    for use with EC2." >> /etc/motd.tail
echo "  Developed by the Berkeley Statistical Computing Facility, February 2014." >> /etc/motd.tail
echo "    Comments and questions can be sent to consult@stat.berkeley.edu." >> /etc/motd.tail
echo "  " >> /etc/motd.tail
echo "  It contains the following computational software," >> /etc/motd.tail
echo "    optimized for statistical computation." >> /etc/motd.tail
echo "  " >> /etc/motd.tail
echo "  * R 3.0.2 linked to OpenBLAS and with a core set of parallel packages:" >> /etc/motd.tail
echo "     (foreach, doParallel, doMPI, Rmpi, pbd)." >> /etc/motd.tail
echo "  * iPython (0.12.1) and Octave (3.2.4)" >> /etc/motd.tail
echo "  * CUDA (5.5) and MAGMA (1.4.1)" >> /etc/motd.tail
echo "  * R and Python packages for GPU computation:" >> /etc/motd.tail
echo "     R's magma package linked to MAGMA 1.3.0" >> /etc/motd.tail
echo "     RCUDA (development version)" >> /etc/motd.tail
echo "     PyCUDA" >> /etc/motd.tail
echo "  " >> /etc/motd.tail


#### Create image ##########################

# 1) now save the image in us-west-2 via point and click on VM page under Actions
# 2) make it public
# 3) test w/ starcluster 0.95.2 I now seem to be able to use StarCluster to start it (formerly it didn't like g2.2xlarge)

# need to put /usr/local/cuda/bin in path as it's not 
# need to copy authorized_keys from either root or ubuntu to paciorek
