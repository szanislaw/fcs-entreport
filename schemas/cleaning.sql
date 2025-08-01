CREATE TABLE IF NOT EXISTS "hskp_service_type" (
  "type_uuid" TEXT,
  "property_uuid" TEXT,
  "type_name" TEXT,
  "type_desc" REAL,
  "type_code" TEXT,
  "duration" INTEGER,
  "credit" INTEGER,
  "inspection" BOOLEAN,
  "inspection_duration" INTEGER,
  "inspection_credit" INTEGER,
  "priority" INTEGER,
  "erequest" BOOLEAN,
  "active" BOOLEAN,
  "created_date" TEXT,
  "created_by" TEXT,
  "modified_date" TEXT,
  "modified_by" TEXT,
  "deleted" TEXT,
  "deleted_date" TEXT,
  "deleted_by" TEXT,
  "attachment" REAL,
  "use_staystatus_credit" BOOLEAN,
  "update_room_status" BOOLEAN,
  "use_location_category_credit" REAL
);

CREATE TABLE IF NOT EXISTS "hskp_matrix_status" (
  "matrix_status_uuid" TEXT,
  "matrix_uuid" TEXT,
  "status" INTEGER,
  "created_date" TEXT,
  "created_by" REAL,
  "modified_date" REAL,
  "modified_by" TEXT,
  "matrix_date" TEXT,
  "co_process_status_id" REAL,
  "co_process_status_name" TEXT,
  "action" TEXT,
  "excludelocation" REAL
);

CREATE TABLE IF NOT EXISTS "hskp_matrix_map_user" (
  "matrix_uuid" TEXT,
  "user_uuid" TEXT
);

CREATE TABLE IF NOT EXISTS "hskp_matrix_map_room_status" (
  "matrix_uuid" TEXT,
  "status_code" TEXT,
  "item_uuid" TEXT
);

CREATE TABLE IF NOT EXISTS "hskp_matrix_detail" (
  "detail_uuid" TEXT,
  "matrix_uuid" TEXT,
  "user_uuid" TEXT,
  "location_uuid" TEXT,
  "credit" REAL,
  "cleaning_uuid" TEXT,
  "sequence" INTEGER,
  "bucket_uuid" REAL,
  "created_date" TEXT,
  "keep_during_stay" BOOLEAN,
  "remark" REAL,
  "latest_batch_update_remark" REAL
);

CREATE TABLE IF NOT EXISTS "hskp_location_indicator_detail" (
  "detail_uuid" TEXT,
  "location_uuid" TEXT,
  "queue_room" TEXT,
  "indicator" REAL,
  "dnd" TEXT,
  "created_date" TEXT,
  "created_by" TEXT,
  "created_from_automation" TEXT,
  "skip_room" TEXT,
  "sleep_out" REAL,
  "green_program" REAL,
  "appear_occupied" REAL,
  "update_pax" REAL,
  "smoking_smell" REAL,
  "update_pax_adult" REAL,
  "update_pax_child" REAL
);

CREATE TABLE IF NOT EXISTS "hskp_location_indicator_audit_trail" (
  "indicator_audit_uuid" TEXT,
  "location_uuid" TEXT,
  "audit_log" TEXT,
  "created_by" TEXT,
  "created_date" TEXT,
  "room_status" REAL,
  "source_type" REAL,
  "request_source" REAL
);

CREATE TABLE IF NOT EXISTS "hskp_location_category_map_tag" (
  "category_uuid" TEXT,
  "tag_uuid" TEXT
);

CREATE TABLE IF NOT EXISTS "hskp_location_category" (
  "category_uuid" TEXT,
  "property_uuid" TEXT,
  "category_name" REAL,
  "category_code" TEXT,
  "credit_base" REAL,
  "credit" REAL,
  "duration" INTEGER,
  "inspection_credit" REAL,
  "inspection_duration" INTEGER,
  "credit2" REAL,
  "duration2" INTEGER,
  "inspection_credit2" REAL,
  "inspection_duration2" INTEGER,
  "rounds" INTEGER,
  "active" BOOLEAN,
  "created_date" TEXT,
  "created_by" TEXT,
  "modified_date" TEXT,
  "modified_by" TEXT,
  "deleted" TEXT,
  "deleted_date" TEXT,
  "deleted_by" TEXT,
  "location_category_uuid" TEXT,
  "location_type" INTEGER
);

CREATE TABLE IF NOT EXISTS "hskp_cleaning_order_map_checklist" (
  "cleaning_checklist_uuid" TEXT,
  "cleaning_uuid" TEXT,
  "checklist_uuid" TEXT,
  "mapping_type" INTEGER,
  "status" INTEGER,
  "created_by" TEXT,
  "created_date" TEXT,
  "modified_by" REAL,
  "modified_date" REAL,
  "deleted" REAL,
  "deleted_by" REAL,
  "deleted_date" REAL,
  "locked" REAL,
  "score" REAL,
  "passing_score" REAL
);

CREATE TABLE IF NOT EXISTS "hskp_cleaning_order_map_additional_task" (
  "cleaning_uuid" TEXT,
  "additional_task_id" INTEGER,
  "status" BOOLEAN
);

CREATE TABLE IF NOT EXISTS "hskp_cleaning_order_inspection" (
  "inspection_uuid" TEXT,
  "cleaning_uuid" TEXT,
  "inspection_by" TEXT,
  "inspection_result" INTEGER,
  "remarks" REAL,
  "rating" INTEGER,
  "completed_by" TEXT,
  "completed_date" TEXT,
  "created_by" TEXT,
  "created_date" TEXT,
  "modified_by" TEXT,
  "modified_date" TEXT,
  "deleted" TEXT,
  "deleted_by" TEXT,
  "deleted_date" TEXT,
  "start_time" REAL,
  "time_spent" REAL,
  "comment" REAL,
  "time_spent_second" REAL
);

CREATE TABLE IF NOT EXISTS "hskp_cleaning_order_detail" (
  "detail_uuid" TEXT,
  "cleaning_uuid" TEXT,
  "status" INTEGER,
  "user_uuid" TEXT,
  "created_by" TEXT,
  "created_date" TEXT,
  "user_type" INTEGER,
  "broadcast" REAL,
  "priority" INTEGER,
  "start_time" TEXT,
  "end_time" REAL,
  "action_type" REAL,
  "room_status" REAL,
  "credit" REAL
);

CREATE TABLE IF NOT EXISTS "hskp_cleaning_order_checklist_detail" (
  "cleaning_chk_det_uuid" TEXT,
  "cleaning_uuid" TEXT,
  "cleaning_checklist_uuid" TEXT,
  "detail_uuid" TEXT,
  "answer" REAL,
  "note" REAL,
  "created_by" TEXT,
  "created_date" TEXT,
  "modified_by" TEXT,
  "modified_date" TEXT,
  "attachment" REAL,
  "optional" BOOLEAN
);

CREATE TABLE IF NOT EXISTS "hskp_cleaning_order" (
  "cleaning_uuid" TEXT,
  "property_uuid" TEXT,
  "cleaning_no" TEXT,
  "location_uuid" TEXT,
  "service_type" TEXT,
  "credit" REAL,
  "duration" INTEGER,
  "remarks" REAL,
  "comments" REAL,
  "status" INTEGER,
  "priority" INTEGER,
  "inspection" BOOLEAN,
  "esignature" BOOLEAN,
  "esignature_key" REAL,
  "queue_room" REAL,
  "start_time" TEXT,
  "notify" INTEGER,
  "notify_count" REAL,
  "reminder_date" REAL,
  "created_by" TEXT,
  "created_date" TEXT,
  "created_name" TEXT,
  "modified_by" TEXT,
  "modified_date" TEXT,
  "completion_remarks" TEXT,
  "job_start" TEXT,
  "job_stop" TEXT,
  "assign_type" INTEGER,
  "assigned_uuid" TEXT,
  "assigned_date" TEXT,
  "assigned_name" TEXT,
  "acknowledged_uuid" TEXT,
  "acknowledged_date" TEXT,
  "acknowledged_name" TEXT,
  "completed_uuid" TEXT,
  "completed_date" TEXT,
  "completed_name" TEXT,
  "inspected_uuid" TEXT,
  "inspected_date" TEXT,
  "inspected_name" TEXT,
  "cancelled_uuid" REAL,
  "cancelled_name" REAL,
  "cancelled_date" REAL,
  "time_spent" REAL,
  "inspection_credit" REAL,
  "inspection_duration" INTEGER,
  "current_deadline" TEXT,
  "original_deadline" TEXT,
  "indicator" REAL,
  "additional_task" TEXT,
  "queued_uuid" TEXT,
  "queued_date" TEXT,
  "queued_name" TEXT,
  "printer_message_uuid" REAL,
  "printer_status_flg" REAL,
  "from_signify" REAL,
  "sequence" REAL,
  "request_source" INTEGER,
  "sleep_out" REAL,
  "new_credit" REAL,
  "appear_occupied" REAL,
  "update_pax" REAL,
  "smoking_smell" REAL
);