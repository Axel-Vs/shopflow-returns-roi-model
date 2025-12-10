# Data Directory

This directory contains the training and test datasets.

## Files

- `ecommerce_returns_train.csv` - Training dataset
- `ecommerce_returns_test.csv` - Test dataset

## Dataset Schema

| Column | Type | Description |
|--------|------|-------------|
| customer_age | int | Age of the customer |
| customer_tenure_days | int | Days since customer registered |
| product_category | str | Category of the product |
| product_price | float | Price of the product |
| days_since_last_purchase | int | Days since last purchase |
| previous_returns | int | Number of previous returns |
| product_rating | float | Product rating (1-5) |
| size_purchased | str | Size purchased (if applicable) |
| discount_applied | float | Discount applied (0-1) |
| is_return | int | Target variable (0=No Return, 1=Return) |

## Data Statistics

Run `enhanced_model.py` to see:
- Training samples count
- Test samples count
- Class distribution
- Feature statistics
