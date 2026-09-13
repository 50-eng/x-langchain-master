from sqlalchemy import Column, String, Integer, Float, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from datetime import datetime

Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    user_id = Column(String(32), primary_key=True)
    username = Column(String(64))
    phone = Column(String(20))
    address = Column(String(255))

class Product(Base):
    __tablename__ = "products"
    product_id = Column(String(32), primary_key=True)
    product_name = Column(String(128))
    price = Column(Float)
    stock = Column(Integer)

class OrderMain(Base):
    __tablename__ = "order_main"
    order_id = Column(String(32), primary_key=True)
    user_id = Column(String(32), ForeignKey("users.user_id"))
    name = Column(String(64))
    phone = Column(String(20))
    address = Column(String(255))
    total_amount = Column(Float)
    status = Column(String(20))  # pending/shipped/delivered/refunded
    create_time = Column(DateTime, default=datetime.now)

class OrderDetail(Base):
    __tablename__ = "order_detail"
    id = Column(Integer, primary_key=True, autoincrement=True)
    order_id = Column(String(32), ForeignKey("order_main.order_id"))
    product_id = Column(String(32), ForeignKey("products.product_id"))
    product_price = Column(Float)
    quantity = Column(Integer)
