#git clone https://github.com/HatScripts/circle-flags.git
#cd circle-flags/flags/
#for file in *.svg; do echo "$file --> $file.png"; qlmanage -t -s 1000 -o . $file; done
#for file in *.svg.png; do echo "white --> transparent:$file"; convert $file -transparent white $file; done
#cd ../../


mkdir world_borders
cd world_borders
curl http://thematicmapping.org/downloads/TM_WORLD_BORDERS_SIMPL-0.3.zip -o world_borders_simplified.zip
curl http://thematicmapping.org/downloads/TM_WORLD_BORDERS-0.3.zip -o world_borders.zip
unzip -o world_borders_simplified.zip
unzip -o world_borders.zip
cd ../

