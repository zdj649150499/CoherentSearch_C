
source ~/bashrc 

date > date.txt

ls *.dat.list | xargs -n 1 -P 8 /tempGPPS/CoherentSearch_C/CoherentSearch_C/bin/coherent_search_list --numharm 32 -t 6.0 --maxdecim 16 --gpu 0 --zaplist Lband.zaplist  
# or
# ls *.dat | xargs -n 1 -P 8 /tempGPPS/CoherentSearch_C/CoherentSearch_C/bin/coherent_search --numharm 32 -t 6.0 --maxdecim 16 --gpu 0 --zaplist Lband.zaplist

python /tempGPPS/CoherentSearch_C/CoherentSearch_C/bin/sift_candidates.py ./ --min-snr 7.0 --top 1000 -o cands.list

date >> date.txt

