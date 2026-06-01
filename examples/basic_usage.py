from pickbuckets import EqualWidthBucket, RareCategoryBucket

ages = EqualWidthBucket(n_bins=3, labels="interval").fit([18, 25, 34, 52, 70])
print(ages.transform([18, 40, 70]))
print(ages.to_json())

countries = RareCategoryBucket(min_frequency=2).fit(["FR", "FR", "US", "DE"])
print(countries.transform(["FR", "US", "CA", None]))

