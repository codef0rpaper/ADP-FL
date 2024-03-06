seed_total=3
current_time=$(date +"%Y%m%d-%H%M%S")
mode=$1
# Run the experiment
for ((i=1; i<=$seed_total; i++))
do
    # no_dp
    python fed_main.py --adaclip --mode $mode --no_dp --seed $i -N 20 --save_path $current_time
    # adp_noise and adp_round
    python fed_main.py --adaclip --mode $mode --adp_round --adp_noise --seed $i -N 20 --save_path $current_time --round_factor 0.99
    # normal noise
    python fed_main.py --adaclip --mode $mode --seed $i -N 20 --save_path $current_time 
    
    # no_dp
    python fed_main.py --adaclip --mode $mode --no_dp --seed $i -N 6 --data prostate --save_path $current_time --epsilonilon 100 --round_threshold 5e-4
    # adp_noise and adp_round
    python fed_main.py --adaclip --mode $mode --adp_round --adp_noise --seed $i -N 6 --data prostate --save_path $current_time --round_factor 0.99 --round_threshold 5e-4 --epsilon 200
    # normal noise
    python fed_main.py --adaclip --mode $mode --seed $i -N 6 --data prostate --save_path $current_time --epsilon 400 --round_threshold 5e-4
    
done
