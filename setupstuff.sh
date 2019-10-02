# creates folders etc
# RUN THIS to setup stuff:
# chmod +x setupstuff.sh
mkdir imgs
mkdir data
cd data
wget https://www2.iis.fraunhofer.de/IPIN/testing/cross.tar.gz
wget https://www2.iis.fraunhofer.de/IPIN/training/horizontal.tar.gz
wget https://www2.iis.fraunhofer.de/IPIN/training/vertical.tar.gz
tar -xvzf cross.tar.gz
tar -xvzf horizontal.tar.gz
tar -xvzf vertical.tar.gz

#chmod +x setupstuff.sh
