import pandas as pd

# load cleaned dataset
df = pd.read_csv("clean_training_dataset.csv")


keywords = {

"architectural":[
"floor","ceiling","door","window","corridor","room","wall","stair",
"elevation","section","finish","layout","partition"
],

"structural":[
"beam","column","foundation","rebar","slab","concrete","steel",
"footing","structural","reinforcement","load","anchor"
],

"mechanical":[
"duct","hvac","fan","air","pump","mechanical","chiller",
"exhaust","ventilation","cooling","heating"
],

"plumbing":[
"pipe","water","drain","toilet","fixture","valve","sanitary",
"sewer","waste","plumbing","sink"
],

"electrical":[
"electrical","panel","receptacle","gfi","conduit","breaker",
"circuit","voltage","lighting","switch","transformer","inverter"
],

"fire_protection":[
"sprinkler","fire","alarm","detector","tamper","pump",
"hydrant","smoke","suppression"
]

}


def extract_features(text):

    text = str(text)

    features = {}

    for category in keywords:

        count = 0

        for word in keywords[category]:
            count += text.count(word)

        features[category] = count

    return pd.Series(features)


feature_df = df["clean_text"].apply(extract_features)

dataset = pd.concat([df, feature_df], axis=1)

dataset.to_csv("feature_dataset.csv", index=False)

print("Feature dataset created")
print(df.info())