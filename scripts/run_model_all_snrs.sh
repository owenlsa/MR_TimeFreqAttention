# Params Settings
py_file="DSF_SE_SC.py"
dataset="RadioML2016.10b"
gpu=$1
lr="1e-3"

use_att="FTA" # "FTA" "CAM,FAM" "CAM,TAM" "CAM,FAM,TAM"
# SNR Settings
all_snrs=0
specific_snrs=(16 14 10 8 4 0)


if [ $all_snrs == 1 ]
then
	echo "Running all SNRs"
	for ((snr=18;snr>=-20;snr-=2))
	do
		echo Running SNR: $snr
		if [ $use_att == 0 ]
		then
			nohup python -u src/$py_file -snr $snr -gpu $gpu -dataset $dataset -lr $lr
		else
			nohup python -u src/$py_file -snr $snr -gpu $gpu -dataset $dataset -lr $lr -att "$use_att"
		fi
	done
else
	echo "Runing specific SNRs"
	for snr in ${specific_snrs[@]}
	do
		echo Runnning SNR: $snr
		if [ $use_att == 0 ]
			then
				nohup python -u src/$py_file -snr $snr -gpu $gpu -dataset $dataset -lr $lr
			else
				nohup python -u src/$py_file -snr $snr -gpu $gpu -dataset $dataset -lr $lr -att "$use_att"
			fi		
	done
fi
