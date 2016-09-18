library(data.table)

# read files
cols = c('Semana', 'Agencia_ID', 'Canal_ID', 'Ruta_SAK', 
         'Cliente_ID', 'Producto_ID')
train = fread('~/projects/kaggle/grupo_bimbo/data/raw/train.csv', 
              select=c(cols, 'Demanda_uni_equil'))
lag = fread(
  '~/projects/kaggle/grupo_bimbo/data/interim/features_demand_lag3.csv')
lag[, index := NULL]
prod_feats = fread(
  '~/projects/kaggle/grupo_bimbo/data/interim/features_producto_tabla.csv')

# merge training data and product features
setkey(train, Producto_ID)
setkey(prod_feats, Producto_ID)
train = prod_feats[train]

# merge on lagged demand
setkey(lag, Producto_ID, Cliente_ID, Semana)
setkey(train, Producto_ID, Cliente_ID, Semana)
train = lag[train]

# clean up train
train = train[Semana > 5]
train[is.na(Demand_uni_equil_lag3), Demand_uni_equil_lag3 := 0]

fwrite(train, '~/projects/kaggle/grupo_bimbo/data/processed/train.csv')

rm(train)
gc()

test = fread('~/projects/kaggle/grupo_bimbo/data/raw/test.csv', 
             select=c(cols, 'id'))

# merge training data and product features
setkey(test, Producto_ID)
setkey(prod_feats, Producto_ID)
test = prod_feats[test]

# merge on lagged demand
setkey(lag, Producto_ID, Cliente_ID, Semana)
setkey(test, Producto_ID, Cliente_ID, Semana)
test = lag[test]

# clean up test
test[is.na(Demand_uni_equil_lag3), Demand_uni_equil_lag3 := 0]

fwrite(test, '~/projects/kaggle/grupo_bimbo/data/processed/test.csv')
