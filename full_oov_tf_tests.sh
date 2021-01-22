CUDA_VISIBLE_DEVICES=0 python retrogan_trainer_attractrepel_working_pytorch.py --fp16 --epochs 150 Data/ft_full/fasttext_seen.hdf Data/ft_full/fasttext_seen_attractrepelretrofitted.hdf ft_full_all_pytorch models/trained_retrogan/ft_full_all_pytorch&
CUDA_VISIBLE_DEVICES=0 python retrogan_trainer_attractrepel_working_pytorch.py --fp16 --epochs 150 Data/ft_ook/fasttext_seen.hdf Data/ft_ook/fasttext_seen_ook_attractrepelretrofitted.hdf ft_ook_all_pytorch models/trained_retrogan/ft_ook_all_pytorch
#local CUDA_VISIBLE_DEVICES=0 python retrogan_trainer_attractrepel_working_pytorch.py --fp16 --epochs 150 Data/glove_full/glove_seen.hdf Data/glove_full/glove_seen_attractrepelretrofitted.hdf glove_full_all_pytorch models/trained_retrogan/glove_full_all_pytorch &
#local CUDA_VISIBLE_DEVICES=1 python retrogan_trainer_attractrepel_working_pytorch.py --fp16 --epochs 150 Data/glove_ook/glove_seen.hdf Data/glove_ook/glove_seen_ook_attractrepelretrofitted.hdf glove_ook_all_pytorch models/trained_retrogan/glove_ook_all_pytorch
CUDA_VISIBLE_DEVICES=0 python retrogan_trainer_attractrepel_working_pytorch.py --fp16 --epochs 150 Data/nb_full/ft_nb_seen.h5 Data/nb_full/nb_retrofitted_attractrepelretrofitted.h5 nb_full_all_pytorch models/trained_retrogan/nb_full_all_pytorch &
CUDA_VISIBLE_DEVICES=0 python retrogan_trainer_attractrepel_working_pytorch.py --fp16 --epochs 150 Data/nb_ook/ft_nb_seen.h5 Data/nb_ook/nb_retrofitted_ook_attractrepel.h5 nb_ook_all_pytorch models/trained_retrogan/nb_ook_all_pytorch



