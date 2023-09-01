python run.py --index nyctaxi --time 30 --interval 48 --dropout 0.5 --lr 1e-4
python run.py --index nyctaxi --time 60 --interval 24 --dropout 0.2 --lr 1e-4
python run.py --index nyctaxi --time 90 --interval 16 --dropout 0.2 --lr 1e-4
python run.py --index nyctaxi --time 60 --interval 24 --pretrained_model_path ./saved_models/index_nyctaxi_30 --dropout 0.2 --lr 1e-4
python run.py --index nyctaxi --time 90 --interval 16 --pretrained_model_path ./saved_models/index_nyctaxi_30 --dropout 0.2 --lr 1e-4
python run.py --index nycbike --time 30 --interval 48 --dropout 0.5 --lr 1e-4
python run.py --index nycbike --time 60 --interval 24 --pretrained_model_path ./saved_models/index_nycbike_30 --dropout 0.2 --lr 1e-4
python run.py --index nycbike --time 90 --interval 16 --pretrained_model_path ./saved_models/index_nycbike_30 --dropout 0.2 --lr 0.5e-4
python run.py --index nycbike --time 60 --interval 24 --dropout 0.2 --lr 1e-4
python run.py --index nycbike --time 90 --interval 16 --dropout 0.2 --lr 0.5e-4