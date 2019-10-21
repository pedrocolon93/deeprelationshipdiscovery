from extract_entities import extract_entities
from knowledge_graph_generation import generate_kg
from scrape_site import scrape_site

if __name__ == '__main__':
    address = input("Give me a web page that you would like to explore").strip()
    # Web page & Cleaned Page
    filename = scrape_site(address,overwrite=True)
    # Entities
    e_list = extract_entities(filename)
    # Discovery
    triples_list = generate_kg(e_list,
                               drd_models_path="./trained_models/deepreldis/2019-04-2314:43:00.000000",
                               target_file_loc='./trained_models/retroembeddings/2019-04-0813:03:02.430691/retroembeddings.h5clean',
                               trained_model_path="./trained_models/retrogans/2019-04-0721:33:44.223104/toretrogen.h5",
                               ft_dir="./fasttext_model/cc.en.300.bin"
                               )
    # Text