import bertalign.utils as utils

def main():
    alignements = utils.read_json("result_dir/lancelot_1/similarities_as_list.json")
    print(alignements[0])



if __name__ == '__main__':
    main()