# 🚚 FleetLogix — Sistema de Gestión Logística Masiva

> Migración de sistemas legacy a una infraestructura de datos moderna con más de **505,000 registros** sintéticos que simulan dos años de operación histórica.

---

## 📋 Tabla de Contenidos

- [Introducción](#introducción)
- [Stack Tecnológico](#stack-tecnológico)
- [Arquitectura de Datos](#arquitectura-de-datos)
- [Modelo Relacional](#modelo-relacional)
- [Reglas de Negocio](#reglas-de-negocio)
- [Generación de Datos Sintéticos](#generación-de-datos-sintéticos)
- [Optimización y Performance](#optimización-y-performance)
- [Data Warehouse en Snowflake](#data-warehouse-en-snowflake)
- [Pipeline ETL](#pipeline-etl)

---

## Introducción

**FleetLogix** es una empresa de logística de gran escala que requería migrar sus sistemas legacy a una infraestructura de datos moderna y robusta.

El objetivo principal fue diseñar un modelo relacional capaz de soportar más de **505,000 registros**, garantizando la integridad referencial y la coherencia temporal de las operaciones.

Como Científico de Datos, lideré la generación de datos sintéticos realistas que simulan dos años de operación histórica, permitiendo el análisis de rutas, mantenimientos y eficiencia de entregas.

---

## Stack Tecnológico

| Capa | Tecnología | Propósito |
|------|-----------|-----------|
| Base de Datos Relacional | PostgreSQL | Modelo transaccional |
| Lenguaje | Python 3.x | Generación y ETL |
| Generación de datos | Faker | Datos de identidad y localización |
| Procesamiento | Pandas & NumPy | Distribuciones estadísticas |
| Conexión BD | Psycopg2 | Batch Insert a PostgreSQL |
| Data Warehouse | Snowflake | Análisis dimensional |
| Orquestación ETL | Python + Schedule | Automatización diaria |

---

## Arquitectura de Datos

El modelo se centra en la entidad **`trips` (Viajes)**, que actúa como el núcleo transaccional conectando:

```
                    ┌─────────────┐
                    │   vehicles  │
                    └──────┬──────┘
                           │
┌─────────────┐    ┌───────▼──────┐    ┌─────────────┐
│   drivers   ├───►│    trips     ├───►│  deliveries │
└─────────────┘    └───────┬──────┘    └─────────────┘
                           │
                    ┌──────▼──────┐    ┌─────────────┐
                    │   routes    │    │ maintenance │
                    └─────────────┘    └─────────────┘
```

| Entidad | Rol en el modelo |
|---------|-----------------|
| `trips` | **Tabla de hechos** — núcleo transaccional |
| `drivers` | Dimensión — recursos humanos |
| `vehicles` | Dimensión — activos físicos |
| `routes` | Dimensión — logística geográfica |
| `deliveries` | Entidad de detalle — depende de trips |
| `maintenance` | Entidad de detalle — depende de vehicles |

---

## Modelo Relacional

### Tablas principales

```sql
-- Vehículos
vehicles (vehicle_id, license_plate, vehicle_type, 
          capacity_kg, fuel_type, acquisition_date, status)

-- Conductores
drivers (driver_id, employee_code, first_name, last_name,
         license_number, license_expiry, phone, hire_date, status)

-- Rutas
routes (route_id, route_code, origin_city, destination_city,
        distance_km, estimated_duration_hours, toll_cost)

-- Viajes (hechos)
trips (trip_id, vehicle_id, driver_id, route_id,
       departure_datetime, arrival_datetime,
       fuel_consumed_liters, total_weight_kg, status)

-- Entregas
deliveries (delivery_id, trip_id, tracking_number, customer_name,
            delivery_address, package_weight_kg,
            scheduled_datetime, delivered_datetime,
            delivery_status, recipient_signature)

-- Mantenimientos
maintenance (maintenance_id, vehicle_id, maintenance_date,
             maintenance_type, description, cost,
             next_maintenance_date, performed_by)
```

### Optimizaciones aplicadas

```sql
-- Índices B-Tree en columnas de filtrado frecuente
CREATE INDEX idx_maintenance_date    ON maintenance (maintenance_date);
CREATE INDEX idx_trips_route_avg_fuel ON trips (route_id, fuel_consumed_liters);
CREATE INDEX idx_trips_driver_id      ON trips (driver_id);
```

---

## Reglas de Negocio

Las siguientes reglas fueron implementadas durante la generación de datos para garantizar consistencia y realismo:

**Consistencia Temporal**
Ningún vehículo puede llegar a su destino antes de su hora de salida:
```
arrival_datetime > departure_datetime
```

**Realismo Operativo**
La distribución de viajes sigue una curva de probabilidad horaria, concentrando la actividad en horarios laborales (no uniforme aleatoria).

**Mantenimiento Preventivo**
Se automatizó la generación de registros de mantenimiento por cada **20 viajes** realizados por vehículo.

**Densidad de Carga**
Cada viaje agrupa entre **2 y 6 entregas** individuales, con una distribución normal centrada en 4 paquetes.

---

## Generación de Datos Sintéticos

### Método `_get_hourly_distribution()`

No genera viajes — define el **comportamiento del tiempo**. Crea una curva de probabilidad para que los viajes sigan patrones de tráfico humano reales, no distribución uniforme.

### Método `generate_trips()`

Transforma las reglas de negocio en datos concretos:

1. **Consulta disponibilidad:** trae vehículos activos, conductores activos y rutas existentes
2. **Selección aleatoria:** elige vehículo, conductor y ruta de las listas disponibles
3. **Aplica distribución horaria:** decide la hora de salida según `_get_hourly_distribution()`
4. **Simula realidad:**
   - Tiempo de viaje: estimado × factor entre `0.8` y `1.3` (simula tráfico/clima)
   - Combustible: distancia × factor de eficiencia del vehículo
   - Carga: entre el `40%` y `90%` de la capacidad total
5. **Estado del viaje:** `completed` si arrival < now, `in_progress` si no

### Carga masiva a PostgreSQL

```python
# Patrón Batch Insert con psycopg2
cursor.executemany(INSERT_QUERY, lista_de_tuplas)
connection.commit()
```

---

## Optimización y Performance

Resultados del análisis con `EXPLAIN ANALYZE` antes y después de agregar índices:

| Query | Descripción | Antes | Después | Mejora |
|-------|-------------|-------|---------|--------|
| Q6 | Mantenimientos 2026 con UPPER y SUBSTRING | 145 ms | 1.42 ms | **~100x** |
| Q7 | Rutas más pesadas (> 500 km acumulados) | 151 ms | 117 ms | 1.3x |
| Q8 | Ciudades con más gasto de combustible | 89.9 ms | 54 ms | 1.7x |
| Q9 | Consumo de ruta vs promedio global (CTE) | 153 ms | 145 ms | 1.1x |
| Q10 | Escalafón de conductores con RANK() | 153 ms | 54 ms | 2.8x |

### Queries de negocio implementadas

```sql
-- Q6: Mantenimientos del año con formato
SELECT UPPER(v.license_plate)         AS placa_mayusculas,
       m.maintenance_date,
       SUBSTRING(m.description, 1, 15) AS resumen_descripcion
FROM maintenance m
JOIN vehicles v ON m.vehicle_id = v.vehicle_id
WHERE m.maintenance_date BETWEEN '2026-01-01' AND '2026-12-31';

-- Q7: Rutas con más de 500 km acumulados
SELECT r.route_code, SUM(r.distance_km) AS total_km
FROM trips t
JOIN routes r ON r.route_id = t.route_id
GROUP BY r.route_code
HAVING SUM(r.distance_km) > 500;

-- Q8: Ciudades con mayor consumo promedio de combustible
SELECT r.destination_city, AVG(t.fuel_consumed_liters) AS avg_fuel
FROM trips t
JOIN routes r ON t.route_id = r.route_id
GROUP BY r.destination_city;

-- Q9: Rutas con consumo mayor al promedio global (CTE)
WITH ResumenRutas AS (
    SELECT r.route_code, AVG(t.fuel_consumed_liters) AS prom_ruta
    FROM trips t
    JOIN routes r ON t.route_id = r.route_id
    GROUP BY r.route_code
)
SELECT * FROM ResumenRutas
WHERE prom_ruta > (SELECT AVG(fuel_consumed_liters) FROM trips);

-- Q10: Escalafón de conductores para bonos (función de ventana)
SELECT d.first_name,
       COUNT(t.trip_id)                              AS viajes,
       RANK() OVER (ORDER BY COUNT(t.trip_id) DESC)  AS puesto
FROM drivers d
JOIN trips t ON d.driver_id = t.driver_id
GROUP BY d.driver_id, d.first_name;
```

---

## Data Warehouse en Snowflake

### Configuración inicial

```sql
USE ROLE ACCOUNTADMIN;
CREATE WAREHOUSE IF NOT EXISTS FLEETLOGIX_WH
  WITH WAREHOUSE_SIZE = 'XSMALL'
  AUTO_SUSPEND = 60
  AUTO_RESUME = TRUE;

CREATE DATABASE IF NOT EXISTS FLEETLOGIX_DW;
USE SCHEMA ANALYTICS;
```

### Modelo Estrella

```
              dim_date
                 │
dim_time ────────┤
                 │
dim_vehicle ─────┼──── fact_deliveries ──── dim_customer
                 │
dim_driver ──────┤
                 │
              dim_route
```

### Features de Snowflake habilitadas

**Time Travel — 30 días**
```sql
ALTER TABLE fact_deliveries SET DATA_RETENTION_TIME_IN_DAYS = 30;

-- Uso: ver datos como estaban hace 1 hora
SELECT * FROM fact_deliveries AT (OFFSET => -3600);
```

**Vistas seguras por rol**
```sql
-- Ventas: solo ve clientes no gubernamentales
CREATE SECURE VIEW v_sales_deliveries AS
SELECT fecha, cliente, peso, status, ingreso
FROM fact_deliveries f
JOIN dim_customer c ON f.customer_key = c.customer_key
WHERE c.customer_type != 'Gobierno';

-- Operaciones: visibilidad completa
CREATE SECURE VIEW v_operations_deliveries AS
SELECT conductor, vehiculo, ruta, retraso, combustible ...
FROM fact_deliveries ...;
```

**Control de acceso**
```sql
CREATE ROLE SALES_ANALYST;
CREATE ROLE OPERATIONS_ANALYST;

GRANT SELECT ON VIEW v_sales_deliveries      TO ROLE SALES_ANALYST;
GRANT SELECT ON VIEW v_operations_deliveries TO ROLE OPERATIONS_ANALYST;
```

---

## Pipeline ETL

Pipeline automatizado en Python que corre diariamente a las **2:00 AM**:

```
PostgreSQL (datos del día anterior)
          │
          │  EXTRACT — query con JOINs sobre trips, drivers, vehicles, routes
          ▼
     DataFrame Pandas
          │
          │  TRANSFORM
          │  · Calcular delivery_time_minutes y delay_minutes
          │  · Calcular fuel_efficiency_km_per_liter
          │  · Calcular cost_per_delivery y revenue_per_delivery
          │  · Validaciones de calidad (pesos, tiempos negativos)
          │  · SCD Tipo 2 para drivers y vehicles
          ▼
     DataFrame limpio
          │
          ├──► load_dimensions() → dim_customer, dim_driver (MERGE)
          │
          └──► load_facts()      → fact_deliveries (batch insert)
                    │
                    ▼
             daily_summary (totales pre-calculados para dashboards)
```

### Métricas del pipeline

```python
self.metrics = {
    'records_extracted':  0,
    'records_transformed': 0,
    'records_loaded':     0,
    'errors':             0
}
```

---

## Autor

Proyecto desarrollado como ejercicio de arquitectura de datos moderna, cubriendo el ciclo completo desde el modelo relacional transaccional hasta el Data Warehouse analítico en la nube.
