"""
Unificación de datasets Olist -> modelo estrella (Power BI) + tablas planas (EDA).
Uso: python unificar_olist.py <carpeta_csv_originales> <carpeta_salida>
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.target import TARGET_COL, build_target

SRC = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
OUT = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("olist_limpio")
OUT.mkdir(parents=True, exist_ok=True)

f = lambda n: SRC / n

# ---------------------------------------------------------------- carga
date_cols = ["order_purchase_timestamp", "order_approved_at",
             "order_delivered_carrier_date", "order_delivered_customer_date",
             "order_estimated_delivery_date"]
orders = pd.read_csv(f("olist_orders_dataset.csv"), parse_dates=date_cols)
items = pd.read_csv(f("olist_order_items_dataset.csv"), parse_dates=["shipping_limit_date"])
pay = pd.read_csv(f("olist_order_payments_dataset.csv"))
rev = pd.read_csv(f("olist_order_reviews_dataset.csv"),
                  parse_dates=["review_creation_date", "review_answer_timestamp"])
cust = pd.read_csv(f("olist_customers_dataset.csv"))
prod = pd.read_csv(f("olist_products_dataset.csv"))
sell = pd.read_csv(f("olist_sellers_dataset.csv"))
geo = pd.read_csv(f("olist_geolocation_dataset.csv"))
trad = pd.read_csv(f("product_category_name_translation.csv"))

# ---------------------------------------------------------------- dim_geolocation
# 1M filas -> 1 fila por zip. Se descartan coordenadas fuera de Brasil y se usa la mediana.
geo = geo[geo.geolocation_lat.between(-34, 6) & geo.geolocation_lng.between(-74, -34)]
dim_geo = (geo.groupby("geolocation_zip_code_prefix")
              .agg(lat=("geolocation_lat", "median"),
                   lng=("geolocation_lng", "median"),
                   ciudad=("geolocation_city", lambda s: s.mode().iat[0]),
                   estado=("geolocation_state", lambda s: s.mode().iat[0]))
              .reset_index().rename(columns={"geolocation_zip_code_prefix": "zip_prefix"}))

# ---------------------------------------------------------------- dim_products
# Dos categorías no están en la tabla de traducción oficial.
extra = pd.DataFrame({
    "product_category_name": ["pc_gamer", "portateis_cozinha_e_preparadores_de_alimentos"],
    "product_category_name_english": ["pc_gamer", "portable_kitchen_food_preparers"]})
trad = pd.concat([trad, extra], ignore_index=True)
dim_prod = (prod.merge(trad, on="product_category_name", how="left")
                .rename(columns={"product_name_lenght": "product_name_length",
                                 "product_description_lenght": "product_description_length"}))
dim_prod["product_category_name_english"] = dim_prod["product_category_name_english"].fillna("unknown")
dim_prod["product_category_name"] = dim_prod["product_category_name"].fillna("desconocida")
dim_prod["product_volume_cm3"] = (dim_prod.product_length_cm * dim_prod.product_height_cm
                                  * dim_prod.product_width_cm)

# ---------------------------------------------------------------- dim_customers / dim_sellers
dim_cust = cust.rename(columns={"customer_zip_code_prefix": "zip_prefix"})
dim_sell = sell.rename(columns={"seller_zip_code_prefix": "zip_prefix"})

# ---------------------------------------------------------------- agregados a nivel orden
pagos_orden = (pay.groupby("order_id")
    .agg(pago_total=("payment_value", "sum"),
         n_metodos_pago=("payment_type", "nunique"),
         cuotas_max=("payment_installments", "max"),
         # método con mayor monto dentro de la orden (no "first", que es arbitrario)
         tipo_pago_principal=("payment_type", lambda s: pay.loc[s.index].sort_values(
             "payment_value", ascending=False).payment_type.iat[0]))
    .reset_index())

# reviews: 1 por orden (la última respondida). Se guarda cuántas tenía.
rev = rev.sort_values(["order_id", "review_answer_timestamp"])
reviews_orden = (rev.groupby("order_id")
    .agg(review_score=("review_score", "last"),
         review_comment_message=("review_comment_message", "last"),
         review_creation_date=("review_creation_date", "last"),
         n_reviews=("review_id", "size"))
    .reset_index())

items_orden = (items.groupby("order_id")
    .agg(n_items=("order_item_id", "size"),
         n_vendedores=("seller_id", "nunique"),
         total_price=("price", "sum"),
         total_freight=("freight_value", "sum"))
    .reset_index())

# ---------------------------------------------------------------- fact_orders (grano: orden)
fo = (orders.merge(cust[["customer_id", "customer_unique_id", "customer_zip_code_prefix",
                         "customer_city", "customer_state"]], on="customer_id", how="left")
            .merge(items_orden, on="order_id", how="left")
            .merge(pagos_orden, on="order_id", how="left")
            .merge(reviews_orden, on="order_id", how="left"))

d = fo
d["dias_aprobacion"] = (d.order_approved_at - d.order_purchase_timestamp).dt.total_seconds() / 86400
d["dias_a_transportista"] = (d.order_delivered_carrier_date - d.order_purchase_timestamp).dt.total_seconds() / 86400
d["dias_entrega_total"] = (d.order_delivered_customer_date - d.order_purchase_timestamp).dt.total_seconds() / 86400
d["dias_entrega_estimada"] = (d.order_estimated_delivery_date - d.order_purchase_timestamp).dt.total_seconds() / 86400
d["dias_retraso"] = (d.order_delivered_customer_date - d.order_estimated_delivery_date).dt.total_seconds() / 86400

def agregar_target(pedidos, orders):
    """Agrega la columna pedido_retrasado a la tabla de órdenes.

    Se descartan los pedidos sin estado "delivered" o sin fecha de entrega o
    estimada, porque en ellos el retraso no se puede observar.
    """
    
    etiquetados = build_target(orders)
    solo_target = etiquetados[["order_id", TARGET_COL]]
    return pedidos.merge(solo_target, on="order_id", how="left")

d = agregar_target(d, orders)  # merge devuelve una tabla nueva: se re-apunta d para que se exporte con el target
fo = d  # merge devuelve una tabla nueva: se re-apunta fo para que se exporte con el target

d["fecha_compra"] = d.order_purchase_timestamp.dt.normalize()
d["tiene_items"] = d.n_items.notna()
d["tiene_review"] = d.review_score.notna()

# ---------------------------------------------------------------- fact_order_items (grano: ítem)
fi = (items.merge(orders[["order_id", "customer_id", "order_status", "order_purchase_timestamp",
                          "order_delivered_customer_date", "order_estimated_delivery_date"]],
                  on="order_id", how="left")
           .merge(cust[["customer_id", "customer_unique_id"]], on="customer_id", how="left"))
fi["monto_item"] = fi.price + fi.freight_value
fi["fecha_compra"] = fi.order_purchase_timestamp.dt.normalize()

# ---------------------------------------------------------------- dim_calendario
rng = pd.date_range(orders.order_purchase_timestamp.min().normalize(),
                    orders.order_purchase_timestamp.max().normalize(), freq="D")
dim_cal = pd.DataFrame({"fecha": rng})
dim_cal["anio"] = dim_cal.fecha.dt.year
dim_cal["mes"] = dim_cal.fecha.dt.month
dim_cal["anio_mes"] = dim_cal.fecha.dt.strftime("%Y-%m")
dim_cal["trimestre"] = dim_cal.fecha.dt.quarter
dim_cal["dia_semana"] = dim_cal.fecha.dt.dayofweek
dim_cal["nombre_dia"] = dim_cal.fecha.dt.day_name()

# ---------------------------------------------------------------- tabla plana para EDA (grano: ítem)
flat = (fi.merge(dim_prod, on="product_id", how="left")
          .merge(dim_sell.rename(columns={"zip_prefix": "seller_zip_prefix"}), on="seller_id", how="left")
          .merge(fo[["order_id", "customer_zip_code_prefix", "customer_city", "customer_state",
                     "review_score", "tiene_review", "dias_entrega_total", "dias_retraso",
                     TARGET_COL, "n_items", "n_vendedores"]], on="order_id", how="left"))
# Columnas de pago NO se llevan a la tabla de ítems (se inflarían al repetirse por ítem).

# ---------------------------------------------------------------- validaciones
assert len(flat) == len(items), "merge duplicó filas en la tabla plana"
assert fo.order_id.is_unique, "fact_orders no es única por orden"
assert dim_prod.product_id.is_unique and dim_cust.customer_id.is_unique and dim_sell.seller_id.is_unique
assert np.isclose(fi.price.sum(), items.price.sum())
assert np.isclose(fo.pago_total.sum(), pay.payment_value.sum()), "los pagos no cuadran"
assert np.isclose(fo.total_price.sum(), items.price.sum()), "los precios no cuadran"

# ---------------------------------------------------------------- export
for nombre, df in {"fact_orders": fo, "fact_order_items": fi, "dim_customers": dim_cust,
                   "dim_sellers": dim_sell, "dim_products": dim_prod, "dim_geolocation": dim_geo,
                   "dim_calendario": dim_cal, "fact_order_payments": pay,
                   "olist_items_flat_EDA": flat}.items():
    df.to_csv(OUT / f"{nombre}.csv", index=False)
    print(f"{nombre:24s} {df.shape}")
print("OK: todas las validaciones pasaron")
