from infra.mysql.mysql import db_session
from infra.mysql.models import User, Product, OrderMain, OrderDetail
from tools.registry import register_tool

@register_tool(name="mysql_tool", category="database", description="MySQL数据库读写工具")
class MySQLTool:
    def query(self, table: str, params: dict) -> dict:
        """查询数据"""
        session = db_session()
        try:
            model_map = {
                "users": User,
                "products": Product,
                "order_main": OrderMain,
                "order_detail": OrderDetail
            }
            model = model_map.get(table)
            if not model:
                return {"error": f"表 {table} 不存在"}

            query = session.query(model)
            for k, v in params.items():
                if hasattr(model, k):
                    query = query.filter(getattr(model, k) == v)
            results = query.all()
            return {"count": len(results), "data": [r.__dict__ for r in results]}
        except Exception as e:
            return {"error": str(e)}
        finally:
            session.close()

    def write(self, table: str, params: dict) -> dict:
        """修改数据（地址、手机号、数量、状态等）"""
        session = db_session()
        try:
            model_map = {
                "users": User,
                "order_main": OrderMain,
                "order_detail": OrderDetail
            }
            model = model_map.get(table)
            if not model:
                return {"error": f"表 {table} 不支持写操作"}

            primary_key = list(params.keys())[0]  # 第一个参数作为主键条件
            primary_value = params.pop(primary_key)
            session.query(model).filter(
                getattr(model, primary_key) == primary_value
            ).update(params)
            session.commit()
            return {"success": True, "updated": params}
        except Exception as e:
            session.rollback()
            return {"error": str(e)}
        finally:
            session.close()
