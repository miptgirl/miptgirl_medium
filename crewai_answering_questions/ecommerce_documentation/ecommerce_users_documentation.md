# ecommerce_db.users Table Overview

The `ecommerce_db.users` table contains essential information about the users of the ecommerce platform. This table is crucial for understanding user demographics and activity status, and it ties into other tables in the `ecommerce_db` database for more comprehensive analyses.

## Columns

| Column Name | Data Type | Description |
|-------------|------------|-------------|
| user_id     | UInt64     | Unique identifier for each user |
| country     | String     | Country of the user |
| is_active   | UInt8      | User activity status (0 = deactivated, 1 = active) |
| age         | UInt64     | Age of the user in years |


## Additional Information
- The `user_id` column is a unique identifier for each user and can be used to join with other tables in the database.
- The `country` column indicates the country where the user resides, which can be useful for regional analysis.
- The `is_active` column shows whether the user is currently active, which can help in filtering active users for certain analyses.
- The `age` column provides the age of the user, which can be used for demographic studies.
