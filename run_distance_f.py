from gene.learn_distance import learn_distance_f_evo


if __name__ == "__main__":
    import json

    with open("config/brax.json") as f:
        config = json.load(f)

    fit = learn_distance_f_evo(config)

    print(sorted(fit, reverse=True)[:3])
