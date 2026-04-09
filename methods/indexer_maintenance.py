#!/usr/bin/python3
# coding=utf-8
# pylint: disable=C,R,E0401,E0611

#   Copyright 2025 EPAM Systems
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

""" Method """

import time

from pylon.core.tools import log  # pylint: disable=E0611,E0401
from pylon.core.tools import web  # pylint: disable=E0611,E0401


class Method:  # pylint: disable=E1101,R0903,W0201
    """
        Method Resource

        self is pointing to current Module instance

        web.method decorator takes zero or one argument: method name
        Note: web.method decorator must be the last decorator (at top)
    """

    @web.method()
    def indexer_migrate(self, connection_str):  # pylint: disable=R0912,R0914,R0915
        """ Run task target """
        try:
            import tasknode_task  # pylint: disable=E0401,C0415
            #
            self.indexer_enable_logging(
                additional_labels={
                    "stream_id": f"task_id:{tasknode_task.id}",
                },
            )
        except:  # pylint: disable=W0702
            pass
        #
        log.info("Index migration task started")
        start_ts = time.time()
        #
        try:

            #
            # ---
            #

            import gc
            import uuid as python_uuid

            import sqlalchemy
            from sqlalchemy.dialects.postgresql import JSON, JSONB, UUID
            from sqlalchemy.orm import Session, relationship
            from sqlalchemy.schema import CreateSchema, DropSchema

            try:
                from sqlalchemy.orm import declarative_base
            except ImportError:
                from sqlalchemy.ext.declarative import declarative_base

            from pgvector.sqlalchemy import Vector


            vector_dimension = None

            JSON_Base = declarative_base()
            JSONB_Base = declarative_base()


            class JSON_BaseModel(JSON_Base):
                __abstract__ = True
                uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=python_uuid.uuid4)


            class JSON_CollectionStore(JSON_BaseModel):
                __tablename__ = "langchain_pg_collection"

                name = sqlalchemy.Column(sqlalchemy.String)
                cmetadata = sqlalchemy.Column(JSON)

                embeddings = relationship(
                    "JSON_EmbeddingStore",
                    back_populates="collection",
                    passive_deletes=True,
                )

                @classmethod
                def get_by_name(cls, session, name):
                    return session.query(cls).filter(cls.name == name).first()

                @classmethod
                def get_or_create(cls, session, name, cmetadata=None):
                    created = False
                    collection = cls.get_by_name(session, name)
                    if collection:
                        return collection, created
                    #
                    collection = cls(name=name, cmetadata=cmetadata)
                    session.add(collection)
                    session.commit()
                    created = True
                    return collection, created


            class JSON_EmbeddingStore(JSON_BaseModel):
                __tablename__ = "langchain_pg_embedding"

                collection_id = sqlalchemy.Column(
                    UUID(as_uuid=True),
                    sqlalchemy.ForeignKey(
                        f"{JSON_CollectionStore.__tablename__}.uuid",
                        ondelete="CASCADE",
                    ),
                )
                collection = relationship(JSON_CollectionStore, back_populates="embeddings")

                embedding = sqlalchemy.Column(Vector(vector_dimension))
                document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
                cmetadata = sqlalchemy.Column(JSON, nullable=True)

                custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)


            class JSONB_BaseModel(JSONB_Base):
                __abstract__ = True
                uuid = sqlalchemy.Column(UUID(as_uuid=True), primary_key=True, default=python_uuid.uuid4)


            class JSONB_CollectionStore(JSONB_BaseModel):
                __tablename__ = "langchain_pg_collection"

                name = sqlalchemy.Column(sqlalchemy.String)
                cmetadata = sqlalchemy.Column(JSON)

                embeddings = relationship(
                    "JSONB_EmbeddingStore",
                    back_populates="collection",
                    passive_deletes=True,
                )

                @classmethod
                def get_by_name(cls, session, name):
                    return session.query(cls).filter(cls.name == name).first()

                @classmethod
                def get_or_create(cls, session, name, cmetadata=None):
                    created = False
                    collection = cls.get_by_name(session, name)
                    if collection:
                        return collection, created
                    #
                    collection = cls(name=name, cmetadata=cmetadata)
                    session.add(collection)
                    session.commit()
                    created = True
                    return collection, created


            class JSONB_EmbeddingStore(JSONB_BaseModel):
                __tablename__ = "langchain_pg_embedding"

                collection_id = sqlalchemy.Column(
                    UUID(as_uuid=True),
                    sqlalchemy.ForeignKey(
                        f"{JSONB_CollectionStore.__tablename__}.uuid",
                        ondelete="CASCADE",
                    ),
                )
                collection = relationship(JSONB_CollectionStore, back_populates="embeddings")

                embedding = sqlalchemy.Column(Vector(vector_dimension))
                document = sqlalchemy.Column(sqlalchemy.String, nullable=True)
                cmetadata = sqlalchemy.Column(JSONB, nullable=True)

                custom_id = sqlalchemy.Column(sqlalchemy.String, nullable=True)

                __table_args__ = (
                    sqlalchemy.Index(
                        "ix_cmetadata_gin",
                        "cmetadata",
                        postgresql_using="gin",
                        postgresql_ops={"cmetadata": "jsonb_path_ops"},
                    ),
                )

            #
            # ---
            #

            common_engine = sqlalchemy.create_engine(url=connection_str)
            with Session(common_engine) as common_session:
                common_collections = common_session.query(JSON_CollectionStore).all()
                #
                for common_collection in common_collections:
                    schema_name = f"ds_{common_collection.name}"
                    #
                    log.info("Processing schema: %s", schema_name)
                    #
                    target_engine = common_engine.execution_options(
                        schema_translate_map={
                            None: schema_name,
                        },
                    )
                    #
                    with Session(target_engine) as target_session:
                        target_session.execute(
                            DropSchema(
                                schema_name,
                                cascade=True,
                                if_exists=True,
                            )
                        )
                        target_session.commit()
                    #
                    with Session(target_engine) as target_session:
                        target_session.execute(
                            CreateSchema(
                                schema_name,
                                if_not_exists=True,
                            )
                        )
                        target_session.commit()
                    #
                    with Session(target_engine) as target_session, target_session.begin():
                        JSONB_Base.metadata.create_all(target_session.get_bind())
                    #
                    with Session(target_engine) as target_session:
                        target_collection = JSONB_CollectionStore(
                            name=common_collection.name,
                            cmetadata=common_collection.cmetadata,
                        )
                        #
                        target_session.add(target_collection)
                        target_session.commit()
                        #
                        target_collection = target_session.query(JSONB_CollectionStore).filter(
                            JSONB_CollectionStore.name == common_collection.name
                        ).first()
                        #
                        objs = []
                        max_objs_per_batch = 500
                        #
                        with Session(common_engine) as common_query_session:
                            common_objs = common_query_session.query(JSON_EmbeddingStore).filter(
                                JSON_EmbeddingStore.collection_id == common_collection.uuid
                            ).yield_per(max_objs_per_batch)
                            #
                            for common_obj in common_objs:
                                objs.append(
                                    JSONB_EmbeddingStore(
                                        embedding=common_obj.embedding,
                                        document=common_obj.document,
                                        cmetadata=common_obj.cmetadata,
                                        custom_id=common_obj.custom_id,
                                        collection_id=target_collection.uuid,
                                    )
                                )
                                #
                                if len(objs) >= max_objs_per_batch:
                                    target_session.bulk_save_objects(objs)
                                    target_session.commit()
                                    #
                                    objs = []
                            #
                            if objs:
                                target_session.bulk_save_objects(objs)
                                target_session.commit()
                                #
                                objs = []
                    #
                    gc.collect()
            #
            # ---
            #

        except:  # pylint: disable=W0702
            log.exception("Got exception, stopping")
        #
        end_ts = time.time()
        log.info("Index migration task ended (duration = %s)", end_ts - start_ts)
