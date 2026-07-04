"""REST API for datasets: import, labeling, quality report (DESIGN.md §4, §8)."""

import uuid

from fastapi import APIRouter, Depends, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel

from app.api.security import require_auth
from app.config import DATA_DIR
from app.datasets import imports, versions

router = APIRouter(prefix="/api", dependencies=[Depends(require_auth)])


class DatasetCreate(BaseModel):
    name: str


class ImagePatch(BaseModel):
    label: str | None = None
    excluded: bool | None = None
    capture_group: str | None = None


def db(request: Request):
    return request.app.state.db


@router.get("/datasets")
def list_datasets(request: Request):
    return db(request).list_datasets()


@router.post("/datasets", status_code=201)
def create_dataset(body: DatasetCreate, request: Request):
    return db(request).create_dataset(uuid.uuid4().hex[:12], body.name.strip() or "untitled")


@router.get("/datasets/{dataset_id}")
def get_dataset(dataset_id: str, request: Request):
    ds = db(request).get_dataset(dataset_id)
    if not ds:
        raise HTTPException(404, "no such dataset")
    return ds


@router.get("/datasets/{dataset_id}/images")
def list_images(dataset_id: str, request: Request):
    if not db(request).get_dataset(dataset_id):
        raise HTTPException(404, "no such dataset")
    return db(request).list_images(dataset_id)


@router.post("/datasets/{dataset_id}/images")
async def upload_images(dataset_id: str, files: list[UploadFile], request: Request):
    if not db(request).get_dataset(dataset_id):
        raise HTTPException(404, "no such dataset")
    added, duplicates, rejected = 0, 0, []
    for f in files:
        data = await f.read()
        try:
            result = imports.import_file(
                db(request), dataset_id, DATA_DIR / "datasets",
                rel_name=f.filename or "unnamed", content_type=f.content_type or "",
                data=data)
        except imports.ImportError_ as e:
            rejected.append(str(e))
            continue
        if result == "added":
            added += 1
        else:
            duplicates += 1
    if added:
        imports.recompute_groups(db(request), dataset_id)
    return {"added": added, "duplicates": duplicates, "rejected": rejected}


@router.get("/datasets/{dataset_id}/report")
def report(dataset_id: str, request: Request):
    if not db(request).get_dataset(dataset_id):
        raise HTTPException(404, "no such dataset")
    return versions.build_report(db(request), dataset_id)


@router.get("/images/{image_id}/file")
def image_file(image_id: str, request: Request):
    img = db(request).get_image(image_id)
    if not img:
        raise HTTPException(404, "no such image")
    return FileResponse(img["path"])


@router.patch("/images/{image_id}")
def patch_image(image_id: str, body: ImagePatch, request: Request):
    img = db(request).get_image(image_id)
    if not img:
        raise HTTPException(404, "no such image")
    fields = body.model_dump(exclude_unset=True)
    if "excluded" in fields:
        fields["excluded"] = int(fields["excluded"])
    if "capture_group" in fields:
        fields["group_source"] = "manual"  # user-set groups survive recompute
    db(request).update_image(image_id, fields)
    return db(request).get_image(image_id)
