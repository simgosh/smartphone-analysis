import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

#loading original dataset
df = pd.read_csv("smartphones_cleaned_v6.csv")
print(df.head())
print(df.isnull().sum())
print(df.info())

#celan the data
df["rating"].fillna((df["rating"].mean()), inplace=True)
df["processor_speed"].fillna((df["processor_speed"].mean()), inplace=True)
df["processor_brand"].fillna("Null", inplace=True)
df["num_cores"].fillna((df["num_cores"].mean()), inplace=True)
df["num_front_cameras"].fillna((df["num_front_cameras"].mean()), inplace=True)
df["battery_capacity"].fillna((df["battery_capacity"].mean()), inplace=True)
df["fast_charging"].fillna((df["fast_charging"].mean()), inplace=True)
df["primary_camera_front"].fillna((df["primary_camera_front"].mean()), inplace=True)
df.drop(["extended_upto", "os"], axis=1, inplace=True)
print(df.isnull().sum())

#price columns from INR to USD Dollars
df["price"] = df["price"] * 0.012
print(df["price"])

print(df["brand_name"].value_counts())

top_brands = df["brand_name"].value_counts().nlargest(15)
print(top_brands)

fig = px.histogram(top_brands,
                   x=top_brands.index,
                   y=top_brands.values,
                   color=top_brands.index,
                   labels={"brand_name":"Brand Name",
                           "sum of count":"Value"},
                   color_discrete_sequence=px.colors.qualitative.Dark2)
fig.show()

brand_by_price = df.groupby("brand_name")["price"].agg("mean")\
                 .nlargest(10).round(1).sort_values(ascending=False)
print(brand_by_price)

model_by_price = df.groupby("model")["price"].agg("sum")\
                 .nlargest(10).round(1).sort_values(ascending=False)
print(model_by_price)

fig1 = px.pie(model_by_price,
              names=model_by_price.index,
              values=model_by_price.values,
              color=model_by_price.index,
              hole=0.2,
              color_discrete_sequence=px.colors.qualitative.Pastel)
fig1.show()


fig2 = px.bar(brand_by_price,
                x=brand_by_price.index,
                y=model_by_price.values,
                labels={"brand_name":"Brand Name",
                         "y":"Values"},
                title="Top 10 Brands by Price" ,        
                color=brand_by_price.index,
                color_discrete_sequence=px.colors.qualitative.Set2)
fig2.show()

#correlation matrix for include all numeric columns
numeric = df.select_dtypes(include="number")
corr_matrix = numeric.corr()
sns.heatmap(corr_matrix,
            annot=True,
            cmap="Spectral")
plt.title("Correlation Matrix for all numeric columns")
plt.xticks(rotation=45)
plt.yticks(rotation=25)
plt.show()


fig3 = px.scatter(df,
                  x="rating",
                  y="price",
                  hover_name="brand_name",
                  color="brand_name")

fig3.show()


sns.countplot(data=df,
              x="brand_name",
              hue="has_5g")
plt.title("5G Support by Brand")
plt.xticks(rotation=45)
plt.xlabel("Brand Name")
plt.ylabel("Count")
plt.legend(title="Has 5G")
plt.show()

fig4 = px.scatter(df,
                  x="price",
                  y="primary_camera_front",
                  hover_data="model",
                  color="brand_name",
                  size="price",
                  color_continuous_scale=px.colors.sequential.Plasma)
fig4.show()


#apple for each model prices details
apple = df[df["brand_name"]=="apple"].groupby("model")["price"].sum().reset_index()\
        .sort_values(by="price", ascending=False)
print(apple)

fig5 = px.bar(apple,
              x="model",
              y="price",
              color="model",
              hover_data="price",
              title="Apple phones for each model",
              labels={"model":"Model",
                      "price":"Price"})
fig5.show()


fig6 = px.scatter(df,
                  x="price",
                  y="screen_size",
                  color="brand_name",
                  size="price",
                  hover_data="model",
                  title="Price vs Screen Size charts",
                  labels={"price":"Price",
                          "screen_size":"Screen Size"},
                  color_discrete_sequence=px.colors.qualitative.Bold)

fig6.show()

plt.figure(figsize=(10, 6))
sns.lineplot(y='battery_capacity', x='rating', data=df, marker='o',color='orange')
plt.title("Relationship Between Battery Capacity vs Rating")
plt.xlabel("Rating")
plt.ylabel("Battery Capacity")
plt.grid(True)
plt.show()

processor_snapdragon = df[df["processor_brand"]=="snapdragon"].groupby("brand_name")["processor_speed"].sum()\
            .reset_index().sort_values(by="processor_speed", ascending=True)
print(processor_snapdragon)

fig7= px.bar(processor_snapdragon,
             x="brand_name",
             y="processor_speed",
             color="brand_name",
             title="Processor Speed by Brands (Snapdragon)",
             labels={"brand_name":"Brand",
                     "processor_speed":"Processor Speed"})
fig7.show()

processor_name = df["processor_brand"].value_counts()
print(processor_name)

bionic = df[df["processor_brand"]=="bionic"].groupby("model")["processor_speed"].sum()\
            .reset_index().sort_values(by="processor_speed", ascending=True)
print(bionic)

fig8 = px.line(bionic,
               x="model",
               y="processor_speed",
               markers=True,
               title="Model vs Processor Speed (Bionic Brand)",
               labels={"model":"Model",
                       "processor_speed":"Processor Speed"})
fig8.show()

fig9 = px.scatter(df,
                  x="resolution_width",
                  y="resolution_height",
                  size="screen_size",
                  trendline="lowess")

fig9.show()



df["fast_charging_available"] = df["fast_charging_available"].map({1:"Yes",
                                                                   0:"No"})
print(df["fast_charging_available"].value_counts())

print(df["fast_charging"])

fig10 = px.scatter(df,
                   x="fast_charging",
                   y="rating",
                   color="rating",
                   hover_data="brand_name",
                   labels={"fast_charging":"Fast Charging",
                           "rating":"Rating"},
                   title="Fast Charging vs Rating")
fig10.show()


fig11 = px.scatter(df,
                   x="fast_charging",
                   y="ram_capacity",
                   color="fast_charging",
                   size="fast_charging",
                   hover_data="brand_name")
fig11.show()


ram_cap = df.groupby("brand_name")["ram_capacity"].mean()\
          .reset_index().sort_values(by="ram_capacity", ascending=False)
print(ram_cap.head(15))

fig12 = px.bar(ram_cap,
               x="brand_name",
               y="ram_capacity",
               color="brand_name")
fig12.show()

print(df[df["brand_name"]=="nokia"].groupby("model")["price"].sum())

ram_capacity = df[df["ram_capacity"]>=15].groupby("model")["price"].sum()\
               .reset_index().sort_values(by="price",
                                          ascending=False)
print(ram_capacity)

fig13 = px.bar(ram_capacity,
               x="model",
               y="price",
               color="model",
               title="Phone Models by RAM Capacity more than 15",
               labels={"model":"Model",
                        "price":"Price"})

fig13.show()


ratings = df[df["rating"]>=89].groupby(["brand_name","model"])[["rating", "price"]].sum()\
          .reset_index().sort_values(by="price", ascending=False)
print(ratings)

fig14 = px.bar(ratings,
               x="model",
               y="price",
               hover_data="rating",
               color="brand_name",
               title="Phone Models with the Most Ratings",
               labels={"model":"Model",
                       "price":"Price"},
                color_discrete_sequence=px.colors.qualitative.Bold_r    
           )

fig14.show()


samsung = df[df["brand_name"]=="samsung"].groupby("model")[["price", "rating"]]\
          .sum().reset_index().sort_values(by="price", ascending=False).head(20)

print(samsung)

fig15 = px.bar(samsung,
               x="model",
               y="price",
               color="model",
               hover_data="rating",
               title="The 20 Most Expensive Models of Samsung",
               labels={"model":"Model",
                       "price":"Price"},
                color_discrete_sequence=px.colors.qualitative.Set3_r)

fig15.show()


expensive_phones = df[df["price"]>2000].groupby(["brand_name","model"])[["price", "ram_capacity",
                                                                          "rating"]].sum()\
                                                                          .reset_index()\
                                                                          .sort_values(by="price", ascending=False)
print(expensive_phones)

fig16 = px.bar(expensive_phones,
               x="model",
               y="price",
               hover_data="ram_capacity",
               color="brand_name",
               labels={"model":"Model",
                       "price":"Price"},
               title="The Most Expensive Phones (more than $2K)")
fig16.show()



cheapest_phones = df[df["price"]<100].groupby(["brand_name","model"])[["price", "ram_capacity",
                                                                          "rating"]].sum()\
                                                                          .reset_index()\
                                                                          .sort_values(by="price", ascending=False)
print(cheapest_phones)

fig17 = px.bar(cheapest_phones,
               x="model",
               y="price",
               hover_data="ram_capacity",
               color="brand_name",
               labels={"model":"Model",
                       "price":"Price"},
               title="The Most Cheapest Phones (less than $100)")
fig17.show()


xiaomi = df[df["brand_name"]=="xiaomi"].groupby("model")[["price", "rating"]]\
        .sum().reset_index().sort_values(by="price", ascending=False).head(20)
print(xiaomi)

fig18 = px.bar(xiaomi,
               x="model",
               y="price",
               color="model",
               hover_data="rating",
               title="The Most Expensive Xiaomi Brand Phones(20)",
               labels={"model":"Model",
                       "price":"Price"},
                color_discrete_sequence=px.colors.qualitative.Set2)

fig18.show()


huawei = df[df["brand_name"]=="huawei"].groupby("model")[["price", "rating"]]\
         .sum().sort_values(by="price", ascending=False).head(10)
print(huawei)


print(df[["primary_camera_rear", "primary_camera_front"]])

fig19 = px.scatter(df,
                   x="primary_camera_rear",
                   y="primary_camera_front",
                   hover_data="ram_capacity",
                   color="brand_name",
                   size="primary_camera_rear",
                   title="Front camera vs Rear Camera Rel.",
                   labels={"primary_camera_rear":"Rear Camera",
                           "primary_camera_front":"Front Camera"},
                    color_discrete_sequence=px.colors.qualitative.Safe_r)
fig19.show()

samsung_vs_apple = df[(df["brand_name"]=="apple") | (df["brand_name"]=="samsung")].groupby("model")[["price",
                                                                                                     "rating",
                                                                                                     "ram_capacity",
                                                                                                     "fast_charging", 
                                                                                                     "battery_capacity"]]\
                                                                                                     .sum().reset_index()\
                                                                                                     .sort_values(by="price", 
                                                                                                                  ascending=False).head(40)
print(samsung_vs_apple)

fig20 = px.bar(samsung_vs_apple,
               x="model",
               y="price",
               hover_data="rating",
               color="model",
               title="Top 30 Phones Apple vs. Samsung Phones by Price")
fig20.show()


huawei_vs_xiaomi = df[(df["brand_name"]=="huawei") | (df["brand_name"]=="xiaomi")].groupby("model")\
                   [["price", "rating"]].sum().reset_index().sort_values(by="price",ascending=False).head(15)

print(huawei_vs_xiaomi)

fig21 = px.bar(huawei_vs_xiaomi,
               x="model",
               y="price",
               hover_data="rating",
               color="model",
               title="Top 15 Phones of Huawei and Xiaomi Brands(by Price)",
               color_discrete_sequence=px.colors.qualitative.Dark24_r)

fig21.show()

processor_comp = df[(df["processor_brand"]=="snapdragon") | (df["processor_brand"] =="helio") | (df["processor_brand"]=="dimensity")]\
                 .groupby(["processor_brand","brand_name"])[["price", "rating", "ram_capacity", "primary_camera_rear", "primary_camera_front"]]\
                 .sum().reset_index().sort_values(by="price", ascending=False).head(25)

print(processor_comp)

fig22 = px.bar(processor_comp,
               x="brand_name",
               y="price",
               color="processor_brand",
               hover_data=["rating", "ram_capacity", "primary_camera_front"],
               title="Comparison of Snapdragon & Helio & Dimensity Processor Brands",
               color_discrete_sequence=px.colors.qualitative.Dark2_r)
fig22.show()

no_5G_apple_phones = df[(df["has_5g"]==False) & (df["processor_brand"]=="bionic")].groupby(["model"])[["price",
                                                                                                        "rating"]].sum().reset_index().sort_values(by="price",
                                                                                                                     ascending=False)

print(no_5G_apple_phones)

yes_5G_apple_phones = df[(df["has_5g"]==True) & (df["processor_brand"]=="bionic")].groupby(["model"])[["price",
                                                                                                        "rating"]].sum().reset_index().sort_values(by="price",
                                                                                                                     ascending=False)

print(yes_5G_apple_phones)

fig23 = px.bar(no_5G_apple_phones,
               x="model",
               y="price",
               hover_data="rating",
               color="model",
               title="NO-5G Apple Phones",
               color_discrete_sequence=px.colors.qualitative.Vivid_r)

fig23.show()

fig24 = px.bar(yes_5G_apple_phones,
               x="model",
               y="price",
               hover_data="rating",
               color="model",
               title="YES-5G Apple Phones",
               color_discrete_sequence=px.colors.qualitative.Vivid)

fig24.show()

fig25 = px.scatter(df,
                  x="resolution_width",
                  y="resolution_height",
                  color="brand_name",
                  size="screen_size",
                  hover_data="screen_size",
                  title="Resolution Width vs. Height Rel.",
                  labels={"resolution_width":"Resolution Width",
                          "resolution_height":"Resolution Height"},
                  color_discrete_sequence=px.colors.qualitative.Plotly_r)

fig25.show()


fig26 = px.scatter(df,
                   x="screen_size",
                   y="price",
                   color="processor_brand",
                   size="screen_size",
                   title="Screen Size vs Price",
                   labels={"screen_size":"Screen Size",
                           "price":"Price"},
                   hover_data="model",
                   color_discrete_sequence=px.colors.qualitative.Plotly)
fig26.show()


#create new categorical column with price for price range
conditions = [
    (df["price"]<=350),
    (df["price"] > 350) & (df["price"]<=1250),
    (df["price"]>1250)
]

values = ["Cheap", "Medium", "Expensive"]

df["price_range"] = np.select(conditions, values)

print(df.head())

fig27 = px.bar(df,
                x="brand_name",
                y="price",
                color="price_range",
                title="Price Range according to Brands",
                labels={"brand_name":"Brand",
                        "price":"Price"},
                color_discrete_sequence=px.colors.qualitative.T10)
fig27.show()



print(df["price_range"].value_counts())

palette = sns.color_palette("Set2")
sns.countplot(data=df,
              x="price_range",
              palette=palette)
plt.title("Count of Price Range")
plt.xlabel("Price Range")
plt.ylabel("Count")
plt.show()


price_ranges = df.groupby("price_range")["rating"].mean().reset_index().sort_values(by="rating",
                                                                                    ascending=False)

print(price_ranges)

price_range_by_brand = df.groupby(["brand_name", "price_range"])["price"].mean()\
                       .reset_index().sort_values(by="price", ascending=False).round(0)
                       

print(price_range_by_brand)

fig28 = px.bar(price_range_by_brand,
               x="brand_name",
               y="price",
               color="price_range",
               title="Brand Name by Price Range",
               labels={"brand_name":"Brand Name",
                       "price":"Price"},
               color_discrete_sequence=px.colors.qualitative.G10_r)

fig28.show()
