
-- ========================
-- 1. Staff Table
-- ========================
CREATE TABLE staff (
    staff_id INTEGER PRIMARY KEY,
    staff_name TEXT NOT NULL,
    position TEXT NOT NULL,
    hire_date DATE NOT NULL,
    employment_status TEXT CHECK (employment_status IN ('Active', 'On Leave', 'Terminated')) DEFAULT 'Active',
    role_category TEXT CHECK (role_category IN ('Housekeeping', 'Front Desk', 'Security', 'Maintenance', 'Supervisor')) DEFAULT 'Housekeeping',
    shift_preference TEXT CHECK (shift_preference IN ('Morning', 'Evening', 'Night')) DEFAULT 'Morning',
    is_present BOOLEAN DEFAULT TRUE
);

-- ========================
-- 2. Shifts Table
-- ========================
CREATE TABLE shifts (
    shift_id INTEGER PRIMARY KEY,
    staff_id INTEGER NOT NULL,
    shift_date DATE NOT NULL,
    shift_type TEXT NOT NULL,
    hotel_id INTEGER,
    shift_supervisor_id INTEGER,
    shift_location TEXT DEFAULT 'Main Building',
    rooms_cleaned INTEGER,
    overtime_hours REAL,
    sick_leave BOOLEAN DEFAULT FALSE,
    reported_for_duty BOOLEAN DEFAULT TRUE,
    late_minutes INTEGER DEFAULT 0,
    incidents_reported TEXT,
    FOREIGN KEY (staff_id) REFERENCES staff(staff_id),
    FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id),
    FOREIGN KEY (shift_supervisor_id) REFERENCES staff(staff_id)
);

-- ========================
-- 3. Room Cleaning Table
-- ========================
CREATE TABLE room_cleaning (
    cleaning_id INTEGER PRIMARY KEY,
    shift_id INTEGER NOT NULL,
    staff_id INTEGER NOT NULL,
    room_id INTEGER NOT NULL,
    cleaning_start TIMESTAMP,
    cleaning_end TIMESTAMP,
    is_rush BOOLEAN DEFAULT FALSE,
    re_clean_required BOOLEAN DEFAULT FALSE,
    cleaning_rating INTEGER CHECK (cleaning_rating BETWEEN 1 AND 5),
    cleaning_type TEXT CHECK (cleaning_type IN ('Standard', 'Deep', 'Turn-down', 'Post-maintenance')),
    supply_issues BOOLEAN DEFAULT FALSE,
    FOREIGN KEY (shift_id) REFERENCES shifts(shift_id),
    FOREIGN KEY (staff_id) REFERENCES staff(staff_id),
    FOREIGN KEY (room_id) REFERENCES rooms(room_id)
);

-- ========================
-- 4. Room Inspections Table
-- ========================
CREATE TABLE room_inspections (
    inspection_id INTEGER PRIMARY KEY,
    cleaning_id INTEGER NOT NULL,
    passed BOOLEAN,
    inspector_id INTEGER NOT NULL,
    inspection_time TIMESTAMP,
    inspector_comments TEXT,
    inspection_type TEXT DEFAULT 'Post-Cleaning',
    FOREIGN KEY (cleaning_id) REFERENCES room_cleaning(cleaning_id),
    FOREIGN KEY (inspector_id) REFERENCES staff(staff_id)
);

-- ========================
-- 5. Lost & Found Table
-- ========================
CREATE TABLE lost_found (
    item_id INTEGER PRIMARY KEY,
    reported_by INTEGER NOT NULL,
    item_description TEXT,
    item_category TEXT,
    report_time TIMESTAMP,
    resolution_time TIMESTAMP,
    guest_claimed BOOLEAN DEFAULT FALSE,
    resolved_by INTEGER NOT NULL,
    notes TEXT,
    FOREIGN KEY (reported_by) REFERENCES staff(staff_id),
    FOREIGN KEY (resolved_by) REFERENCES staff(staff_id)
);

-- ========================
-- 6. Guest Complaints Table
-- ========================
CREATE TABLE guest_complaints (
    complaint_id INTEGER PRIMARY KEY,
    room_id INTEGER NOT NULL,
    complaint_time TIMESTAMP,
    resolved_time TIMESTAMP,
    description TEXT,
    complaint_source TEXT CHECK (complaint_source IN ('Phone', 'App', 'Front Desk')),
    severity TEXT CHECK (severity IN ('Low', 'Medium', 'High')),
    follow_up_required BOOLEAN DEFAULT FALSE,
    resolved_by INTEGER NOT NULL,
    FOREIGN KEY (room_id) REFERENCES rooms(room_id),
    FOREIGN KEY (resolved_by) REFERENCES staff(staff_id)
);

-- ========================
-- 7. Training Table
-- ========================
CREATE TABLE training (
    training_id INTEGER PRIMARY KEY,
    staff_id INTEGER NOT NULL,
    training_date DATE,
    hours REAL,
    topic TEXT,
    provider TEXT,
    assessment_score REAL CHECK (assessment_score BETWEEN 0 AND 100),
    FOREIGN KEY (staff_id) REFERENCES staff(staff_id)
);

-- ========================
-- 8. Roster Table
-- ========================
CREATE TABLE roster (
    roster_id INTEGER PRIMARY KEY,
    staff_id INTEGER NOT NULL,
    scheduled_date DATE,
    scheduled_hours INTEGER,
    actual_hours INTEGER,
    shift_type TEXT CHECK (shift_type IN ('Morning', 'Evening', 'Night')),
    hotel_id INTEGER,
    attendance_status TEXT CHECK (attendance_status IN ('Present', 'Absent', 'Late', 'Excused')) DEFAULT 'Present',
    notes TEXT,
    FOREIGN KEY (staff_id) REFERENCES staff(staff_id),
    FOREIGN KEY (hotel_id) REFERENCES hotels(hotel_id)
);

-- ========================
-- 9. Staff Presence Log
-- ========================
CREATE TABLE staff_presence_log (
    log_id INTEGER PRIMARY KEY,
    staff_id INTEGER NOT NULL,
    log_date DATE NOT NULL,
    present BOOLEAN DEFAULT TRUE,
    reason TEXT,
    FOREIGN KEY (staff_id) REFERENCES staff(staff_id)
);

-- ========================
-- 10. Staff Incidents
-- ========================
CREATE TABLE staff_incidents (
    incident_id INTEGER PRIMARY KEY,
    staff_id INTEGER NOT NULL,
    incident_date DATE NOT NULL,
    description TEXT,
    severity TEXT CHECK (severity IN ('Low', 'Moderate', 'High')) DEFAULT 'Low',
    action_taken TEXT,
    FOREIGN KEY (staff_id) REFERENCES staff(staff_id)
);

-- ========================
-- 11. Rooms Table (Schema Patch)
-- ========================
CREATE TABLE IF NOT EXISTS rooms (
    room_id INTEGER PRIMARY KEY,
    room_number TEXT NOT NULL,
    room_type TEXT CHECK (room_type IN ('Single', 'Double', 'Suite', 'Deluxe')) DEFAULT 'Single',
    is_available BOOLEAN DEFAULT TRUE
);


-- Staff Entries

INSERT INTO staff (staff_id, staff_name, position, hire_date, employment_status, role_category, shift_preference, is_present)
VALUES (1, 'Jesse Moore', 'Technician', 
        '2024-01-07', 
        'Terminated', 
        'Security', 
        'Night', 
        1);


INSERT INTO staff (staff_id, staff_name, position, hire_date, employment_status, role_category, shift_preference, is_present)
VALUES (2, 'Tracie Young', 'Technician', 
        '2022-11-20', 
        'Active', 
        'Maintenance', 
        'Morning', 
        0);


INSERT INTO staff (staff_id, staff_name, position, hire_date, employment_status, role_category, shift_preference, is_present)
VALUES (3, 'Zachary Young', 'Supervisor', 
        '2025-01-26', 
        'Active', 
        'Security', 
        'Night', 
        0);


INSERT INTO staff (staff_id, staff_name, position, hire_date, employment_status, role_category, shift_preference, is_present)
VALUES (4, 'Tony Johnson', 'Technician', 
        '2025-05-01', 
        'On Leave', 
        'Front Desk', 
        'Night', 
        1);


INSERT INTO staff (staff_id, staff_name, position, hire_date, employment_status, role_category, shift_preference, is_present)
VALUES (5, 'Michael Curry', 'Supervisor', 
        '2022-10-07', 
        'On Leave', 
        'Housekeeping', 
        'Evening', 
        0);


INSERT INTO staff (staff_id, staff_name, position, hire_date, employment_status, role_category, shift_preference, is_present)
VALUES (6, 'Joseph Nguyen', 'Technician', 
        '2022-11-02', 
        'Active', 
        'Front Desk', 
        'Morning', 
        0);


INSERT INTO staff (staff_id, staff_name, position, hire_date, employment_status, role_category, shift_preference, is_present)
VALUES (7, 'Susan Anthony', 'Housekeeper', 
        '2022-11-30', 
        'Terminated', 
        'Security', 
        'Evening', 
        0);


INSERT INTO staff (staff_id, staff_name, position, hire_date, employment_status, role_category, shift_preference, is_present)
VALUES (8, 'Donald Anderson', 'Housekeeper', 
        '2024-12-04', 
        'Terminated', 
        'Security', 
        'Evening', 
        0);


INSERT INTO staff (staff_id, staff_name, position, hire_date, employment_status, role_category, shift_preference, is_present)
VALUES (9, 'Jimmy Johnson', 'Technician', 
        '2024-06-13', 
        'Active', 
        'Front Desk', 
        'Night', 
        0);


INSERT INTO staff (staff_id, staff_name, position, hire_date, employment_status, role_category, shift_preference, is_present)
VALUES (10, 'Julia Mathis', 'Housekeeper', 
        '2025-05-29', 
        'Active', 
        'Maintenance', 
        'Evening', 
        1);

-- Shifts Entries

INSERT INTO shifts (shift_id, staff_id, shift_date, shift_type, hotel_id, shift_supervisor_id, shift_location,
                    rooms_cleaned, overtime_hours, sick_leave, reported_for_duty, late_minutes, incidents_reported)
VALUES (1, 4, '2025-07-14',
        'Evening', 2, 5,
        'Main Building', 9, 
        2.49, 0, 0, 
        8, 'Point door feeling into create usually.');


INSERT INTO shifts (shift_id, staff_id, shift_date, shift_type, hotel_id, shift_supervisor_id, shift_location,
                    rooms_cleaned, overtime_hours, sick_leave, reported_for_duty, late_minutes, incidents_reported)
VALUES (2, 4, '2025-07-09',
        'Morning', 2, 3,
        'East Wing', 10, 
        3.67, 0, 1, 
        10, 'Couple firm environment give five beat.');


INSERT INTO shifts (shift_id, staff_id, shift_date, shift_type, hotel_id, shift_supervisor_id, shift_location,
                    rooms_cleaned, overtime_hours, sick_leave, reported_for_duty, late_minutes, incidents_reported)
VALUES (3, 8, '2025-07-18',
        'Morning', 4, 10,
        'Tower 2', 13, 
        3.98, 0, 1, 
        8, 'Deep some small move surface.');


INSERT INTO shifts (shift_id, staff_id, shift_date, shift_type, hotel_id, shift_supervisor_id, shift_location,
                    rooms_cleaned, overtime_hours, sick_leave, reported_for_duty, late_minutes, incidents_reported)
VALUES (4, 7, '2025-07-02',
        'Morning', 2, 2,
        'East Wing', 6, 
        2.26, 1, 0, 
        2, 'Find take stand large former.');


INSERT INTO shifts (shift_id, staff_id, shift_date, shift_type, hotel_id, shift_supervisor_id, shift_location,
                    rooms_cleaned, overtime_hours, sick_leave, reported_for_duty, late_minutes, incidents_reported)
VALUES (5, 7, '2025-07-12',
        'Evening', 4, 6,
        'Main Building', 10, 
        2.76, 1, 1, 
        3, 'Father talk will available Congress reason concern.');


INSERT INTO shifts (shift_id, staff_id, shift_date, shift_type, hotel_id, shift_supervisor_id, shift_location,
                    rooms_cleaned, overtime_hours, sick_leave, reported_for_duty, late_minutes, incidents_reported)
VALUES (6, 9, '2025-06-30',
        'Morning', 2, 7,
        'Tower 2', 5, 
        2.63, 1, 1, 
        13, 'Company finish interesting occur investment.');


INSERT INTO shifts (shift_id, staff_id, shift_date, shift_type, hotel_id, shift_supervisor_id, shift_location,
                    rooms_cleaned, overtime_hours, sick_leave, reported_for_duty, late_minutes, incidents_reported)
VALUES (7, 4, '2025-07-25',
        'Morning', 1, 9,
        'Tower 2', 9, 
        0.32, 1, 0, 
        12, 'Somebody able trade per by gas piece community.');


INSERT INTO shifts (shift_id, staff_id, shift_date, shift_type, hotel_id, shift_supervisor_id, shift_location,
                    rooms_cleaned, overtime_hours, sick_leave, reported_for_duty, late_minutes, incidents_reported)
VALUES (8, 8, '2025-07-05',
        'Morning', 4, 7,
        'Main Building', 12, 
        0.62, 0, 1, 
        7, 'Business summer shake form.');


INSERT INTO shifts (shift_id, staff_id, shift_date, shift_type, hotel_id, shift_supervisor_id, shift_location,
                    rooms_cleaned, overtime_hours, sick_leave, reported_for_duty, late_minutes, incidents_reported)
VALUES (9, 4, '2025-06-29',
        'Evening', 3, 7,
        'Main Building', 10, 
        0.01, 0, 1, 
        13, 'Which everybody I social.');


INSERT INTO shifts (shift_id, staff_id, shift_date, shift_type, hotel_id, shift_supervisor_id, shift_location,
                    rooms_cleaned, overtime_hours, sick_leave, reported_for_duty, late_minutes, incidents_reported)
VALUES (10, 5, '2025-07-13',
        'Evening', 3, 4,
        'Main Building', 14, 
        3.48, 1, 1, 
        8, 'More Congress eight difference four rate seat get.');

-- Room Cleaning Entries

-- ========================
-- Room Cleaning Entries (With staff_id)
-- ========================
INSERT INTO room_cleaning (
    cleaning_id, shift_id, staff_id, room_id, cleaning_start, cleaning_end,
    is_rush, re_clean_required, cleaning_rating, cleaning_type, supply_issues
)
VALUES
(1, 6, 9, 1, '2025-07-01 09:00:00', '2025-07-01 09:45:00', 0, 0, 5, 'Standard', 0),
(2, 7, 4, 2, '2025-07-03 11:00:00', '2025-07-03 11:40:00', 1, 1, 4, 'Deep', 1),
(3, 3, 8, 3, '2025-07-02 13:15:00', '2025-07-02 14:00:00', 0, 0, 3, 'Post-maintenance', 0),
(4, 4, 7, 4, '2025-07-04 08:30:00', '2025-07-04 09:20:00', 0, 1, 2, 'Standard', 1),
(5, 10, 5, 5, '2025-07-05 17:45:00', '2025-07-05 18:30:00', 0, 0, 5, 'Turn-down', 0),
(6, 2, 4, 6, '2025-07-06 07:00:00', '2025-07-06 07:50:00', 1, 0, 4, 'Standard', 1),
(7, 8, 8, 7, '2025-07-07 10:10:00', '2025-07-07 10:40:00', 0, 0, 3, 'Deep', 0),
(8, 6, 9, 8, '2025-07-08 12:00:00', '2025-07-08 12:30:00', 1, 1, 2, 'Post-maintenance', 1),
(9, 1, 4, 9, '2025-07-09 09:00:00', '2025-07-09 09:30:00', 0, 0, 4, 'Turn-down', 0),
(10, 9, 4, 10, '2025-07-10 15:00:00', '2025-07-10 15:45:00', 0, 0, 5, 'Standard', 0);

-- Room Inspections Entries

INSERT INTO room_inspections (inspection_id, cleaning_id, passed, inspector_id, inspection_time, inspector_comments, inspection_type)
VALUES (1, 1, 1, 7, '2025-07-22 16:53:34',
        'Available watch create front even tend.', 'Follow-up');


INSERT INTO room_inspections (inspection_id, cleaning_id, passed, inspector_id, inspection_time, inspector_comments, inspection_type)
VALUES (2, 2, 1, 7, '2025-07-24 19:42:03',
        'Provide design system so challenge.', 'Complaint');


INSERT INTO room_inspections (inspection_id, cleaning_id, passed, inspector_id, inspection_time, inspector_comments, inspection_type)
VALUES (3, 3, 0, 4, '2025-07-20 19:15:58',
        'General society term daughter.', 'Follow-up');


INSERT INTO room_inspections (inspection_id, cleaning_id, passed, inspector_id, inspection_time, inspector_comments, inspection_type)
VALUES (4, 4, 0, 7, '2025-07-27 00:22:17',
        'Interest consumer media president.', 'Post-Cleaning');


INSERT INTO room_inspections (inspection_id, cleaning_id, passed, inspector_id, inspection_time, inspector_comments, inspection_type)
VALUES (5, 5, 1, 5, '2025-07-23 09:57:24',
        'Last how movie past seem account.', 'Complaint');


INSERT INTO room_inspections (inspection_id, cleaning_id, passed, inspector_id, inspection_time, inspector_comments, inspection_type)
VALUES (6, 6, 1, 3, '2025-07-18 14:43:06',
        'End bar speech thing tough.', 'Complaint');


INSERT INTO room_inspections (inspection_id, cleaning_id, passed, inspector_id, inspection_time, inspector_comments, inspection_type)
VALUES (7, 7, 1, 3, '2025-07-19 20:18:07',
        'When argue the radio there.', 'Post-Cleaning');


INSERT INTO room_inspections (inspection_id, cleaning_id, passed, inspector_id, inspection_time, inspector_comments, inspection_type)
VALUES (8, 8, 0, 5, '2025-07-20 12:21:54',
        'Pull important image heavy answer for.', 'Complaint');


INSERT INTO room_inspections (inspection_id, cleaning_id, passed, inspector_id, inspection_time, inspector_comments, inspection_type)
VALUES (9, 9, 0, 4, '2025-07-14 10:11:48',
        'Will behind here out artist east out.', 'Complaint');


INSERT INTO room_inspections (inspection_id, cleaning_id, passed, inspector_id, inspection_time, inspector_comments, inspection_type)
VALUES (10, 10, 0, 3, '2025-07-21 16:56:49',
        'Rise perform expect reduce compare military admit.', 'Complaint');

-- Lost & Found Entries

INSERT INTO lost_found (item_id, reported_by, item_description, item_category, report_time, resolution_time, guest_claimed, resolved_by, notes)
VALUES (1, 4, 'Me visit company place.', 
        'Electronics', 
        '2025-07-18 03:05:39', '2025-07-18 05:05:39', 1, 
        9, 'Could believe far myself information answer.');


INSERT INTO lost_found (item_id, reported_by, item_description, item_category, report_time, resolution_time, guest_claimed, resolved_by, notes)
VALUES (2, 10, 'Court themselves interview.', 
        'Electronics', 
        '2025-07-14 14:31:11', '2025-07-15 14:31:11', 0, 
        10, 'Else sing air doctor because democratic very.');


INSERT INTO lost_found (item_id, reported_by, item_description, item_category, report_time, resolution_time, guest_claimed, resolved_by, notes)
VALUES (3, 9, 'Story do factor.', 
        'Clothing', 
        '2025-07-24 17:42:40', '2025-07-25 12:42:40', 1, 
        6, 'Your lawyer and foot special.');


INSERT INTO lost_found (item_id, reported_by, item_description, item_category, report_time, resolution_time, guest_claimed, resolved_by, notes)
VALUES (4, 9, 'Must child political fly.', 
        'Electronics', 
        '2025-07-24 09:25:54', '2025-07-25 17:25:54', 0, 
        2, 'Safe traditional history determine state.');


INSERT INTO lost_found (item_id, reported_by, item_description, item_category, report_time, resolution_time, guest_claimed, resolved_by, notes)
VALUES (5, 4, 'Body account lead.', 
        'Electronics', 
        '2025-07-13 22:39:26', '2025-07-15 00:39:26', 1, 
        10, 'We herself certainly fight mean.');


INSERT INTO lost_found (item_id, reported_by, item_description, item_category, report_time, resolution_time, guest_claimed, resolved_by, notes)
VALUES (6, 1, 'Win movie.', 
        'Jewelry', 
        '2025-07-20 02:41:28', '2025-07-20 12:41:28', 1, 
        9, 'Order between carry she.');


INSERT INTO lost_found (item_id, reported_by, item_description, item_category, report_time, resolution_time, guest_claimed, resolved_by, notes)
VALUES (7, 4, 'Group place training.', 
        'Jewelry', 
        '2025-07-14 17:24:11', '2025-07-15 17:24:11', 1, 
        6, 'Remember along a rich wrong.');


INSERT INTO lost_found (item_id, reported_by, item_description, item_category, report_time, resolution_time, guest_claimed, resolved_by, notes)
VALUES (8, 10, 'Finally talk stay weight may.', 
        'Electronics', 
        '2025-07-22 13:00:29', '2025-07-23 22:00:29', 1, 
        7, 'People whatever case example.');


INSERT INTO lost_found (item_id, reported_by, item_description, item_category, report_time, resolution_time, guest_claimed, resolved_by, notes)
VALUES (9, 4, 'Item full start.', 
        'Electronics', 
        '2025-07-21 11:38:06', '2025-07-22 03:38:06', 1, 
        3, 'Put American effect.');


INSERT INTO lost_found (item_id, reported_by, item_description, item_category, report_time, resolution_time, guest_claimed, resolved_by, notes)
VALUES (10, 5, 'Need though own.', 
        'Electronics', 
        '2025-07-22 09:35:18', '2025-07-23 15:35:18', 0, 
        8, 'Prove sound throw remain audience.');

-- Guest Complaints Entries

INSERT INTO guest_complaints (complaint_id, room_id, complaint_time, resolved_time, description, complaint_source, severity, follow_up_required, resolved_by)
VALUES (1, 9, '2025-07-19 00:32:05', '2025-07-19 05:32:05', 
        'Citizen stop edge natural attention include.', 'App',
        'Medium', 1, 1);


INSERT INTO guest_complaints (complaint_id, room_id, complaint_time, resolved_time, description, complaint_source, severity, follow_up_required, resolved_by)
VALUES (2, 17, '2025-07-19 16:26:07', '2025-07-19 21:26:07', 
        'Government nation too protect.', 'App',
        'High', 1, 4);


INSERT INTO guest_complaints (complaint_id, room_id, complaint_time, resolved_time, description, complaint_source, severity, follow_up_required, resolved_by)
VALUES (3, 11, '2025-07-22 16:19:07', '2025-07-22 21:19:07', 
        'Myself them worker difficult story show.', 'Front Desk',
        'High', 1, 4);


INSERT INTO guest_complaints (complaint_id, room_id, complaint_time, resolved_time, description, complaint_source, severity, follow_up_required, resolved_by)
VALUES (4, 20, '2025-07-25 21:26:58', '2025-07-26 07:26:58', 
        'Organization method civil either share defense.', 'Phone',
        'Low', 1, 2);


INSERT INTO guest_complaints (complaint_id, room_id, complaint_time, resolved_time, description, complaint_source, severity, follow_up_required, resolved_by)
VALUES (5, 5, '2025-07-14 20:13:49', '2025-07-14 22:13:49', 
        'By coach ask these organization.', 'Front Desk',
        'Medium', 1, 2);


INSERT INTO guest_complaints (complaint_id, room_id, complaint_time, resolved_time, description, complaint_source, severity, follow_up_required, resolved_by)
VALUES (6, 2, '2025-07-20 20:40:36', '2025-07-21 08:40:36', 
        'Southern read clearly long guy hard soldier.', 'Front Desk',
        'Medium', 1, 5);


INSERT INTO guest_complaints (complaint_id, room_id, complaint_time, resolved_time, description, complaint_source, severity, follow_up_required, resolved_by)
VALUES (7, 2, '2025-07-26 06:26:49', '2025-07-26 11:26:49', 
        'American send morning while its likely.', 'App',
        'Medium', 1, 5);


INSERT INTO guest_complaints (complaint_id, room_id, complaint_time, resolved_time, description, complaint_source, severity, follow_up_required, resolved_by)
VALUES (8, 5, '2025-07-16 11:28:22', '2025-07-16 22:28:22', 
        'It card result policy question book find in.', 'Phone',
        'Medium', 1, 7);


INSERT INTO guest_complaints (complaint_id, room_id, complaint_time, resolved_time, description, complaint_source, severity, follow_up_required, resolved_by)
VALUES (9, 12, '2025-07-14 17:39:22', '2025-07-14 23:39:22', 
        'Include series enjoy ago couple south.', 'Phone',
        'High', 0, 3);


INSERT INTO guest_complaints (complaint_id, room_id, complaint_time, resolved_time, description, complaint_source, severity, follow_up_required, resolved_by)
VALUES (10, 7, '2025-07-17 18:05:07', '2025-07-17 21:05:07', 
        'Put move life federal.', 'Phone',
        'Low', 1, 6);

-- Training Entries

INSERT INTO training (training_id, staff_id, training_date, hours, topic, provider, assessment_score)
VALUES (1, 10, '2025-05-05',
        1.6, 'According Training', 
        'Oconnor and Sons', 83.2);


INSERT INTO training (training_id, staff_id, training_date, hours, topic, provider, assessment_score)
VALUES (2, 9, '2025-05-15',
        2.4, 'Whole Training', 
        'Anderson, Lopez and Richardson', 95.9);


INSERT INTO training (training_id, staff_id, training_date, hours, topic, provider, assessment_score)
VALUES (3, 5, '2025-07-06',
        3.0, 'Us Training', 
        'Oliver, Johnson and Fry', 91.0);


INSERT INTO training (training_id, staff_id, training_date, hours, topic, provider, assessment_score)
VALUES (4, 7, '2025-07-22',
        1.3, 'Win Training', 
        'Martinez, Reed and Wheeler', 97.7);


INSERT INTO training (training_id, staff_id, training_date, hours, topic, provider, assessment_score)
VALUES (5, 10, '2025-05-06',
        2.6, 'Majority Training', 
        'Peck, Rice and Nelson', 79.5);


INSERT INTO training (training_id, staff_id, training_date, hours, topic, provider, assessment_score)
VALUES (6, 4, '2025-07-06',
        3.6, 'Drive Training', 
        'Williams, Banks and Buckley', 73.1);


INSERT INTO training (training_id, staff_id, training_date, hours, topic, provider, assessment_score)
VALUES (7, 8, '2025-07-14',
        2.5, 'Part Training', 
        'Castillo-Davis', 82.3);


INSERT INTO training (training_id, staff_id, training_date, hours, topic, provider, assessment_score)
VALUES (8, 4, '2025-06-13',
        4.0, 'Leave Training', 
        'Barnett, Blackwell and Johnson', 77.8);


INSERT INTO training (training_id, staff_id, training_date, hours, topic, provider, assessment_score)
VALUES (9, 3, '2025-06-26',
        3.4, 'Sit Training', 
        'Navarro Ltd', 99.7);


INSERT INTO training (training_id, staff_id, training_date, hours, topic, provider, assessment_score)
VALUES (10, 2, '2025-07-13',
        3.2, 'Present Training', 
        'Quinn-Bennett', 99.8);

-- Roster Entries

INSERT INTO roster (roster_id, staff_id, scheduled_date, scheduled_hours, actual_hours, shift_type, hotel_id, attendance_status, notes)
VALUES (1, 8, '2025-07-01',
        8, 9, 'Evening', 
        2, 'Excused', 'Provide admit live college including financial.');


INSERT INTO roster (roster_id, staff_id, scheduled_date, scheduled_hours, actual_hours, shift_type, hotel_id, attendance_status, notes)
VALUES (2, 9, '2025-07-04',
        8, 7, 'Morning', 
        5, 'Absent', 'Purpose watch cup wrong.');


INSERT INTO roster (roster_id, staff_id, scheduled_date, scheduled_hours, actual_hours, shift_type, hotel_id, attendance_status, notes)
VALUES (3, 9, '2025-07-01',
        8, 9, 'Morning', 
        2, 'Excused', 'Serve nation control choice address.');


INSERT INTO roster (roster_id, staff_id, scheduled_date, scheduled_hours, actual_hours, shift_type, hotel_id, attendance_status, notes)
VALUES (4, 9, '2025-07-04',
        8, 6, 'Evening', 
        2, 'Late', 'Present either performance short forward company vote always.');


INSERT INTO roster (roster_id, staff_id, scheduled_date, scheduled_hours, actual_hours, shift_type, hotel_id, attendance_status, notes)
VALUES (5, 5, '2025-07-07',
        8, 9, 'Morning', 
        1, 'Excused', 'Sell court impact TV.');


INSERT INTO roster (roster_id, staff_id, scheduled_date, scheduled_hours, actual_hours, shift_type, hotel_id, attendance_status, notes)
VALUES (6, 6, '2025-07-07',
        8, 8, 'Evening', 
        2, 'Late', 'Degree she reveal street before.');


INSERT INTO roster (roster_id, staff_id, scheduled_date, scheduled_hours, actual_hours, shift_type, hotel_id, attendance_status, notes)
VALUES (7, 5, '2025-07-23',
        8, 10, 'Night', 
        3, 'Absent', 'Now people should.');


INSERT INTO roster (roster_id, staff_id, scheduled_date, scheduled_hours, actual_hours, shift_type, hotel_id, attendance_status, notes)
VALUES (8, 4, '2025-07-02',
        8, 9, 'Morning', 
        1, 'Late', 'Me leader easy conference sure particular important himself.');


INSERT INTO roster (roster_id, staff_id, scheduled_date, scheduled_hours, actual_hours, shift_type, hotel_id, attendance_status, notes)
VALUES (9, 7, '2025-07-24',
        8, 7, 'Night', 
        1, 'Absent', 'Message capital collection Republican against feeling front.');


INSERT INTO roster (roster_id, staff_id, scheduled_date, scheduled_hours, actual_hours, shift_type, hotel_id, attendance_status, notes)
VALUES (10, 8, '2025-07-13',
        8, 9, 'Evening', 
        5, 'Absent', 'Score fact training cultural.');

-- Staff Presence Log Entries

INSERT INTO staff_presence_log (log_id, staff_id, log_date, present, reason)
VALUES (1, 2, '2025-07-15',
        0, 'Particular identify loss hold move.');


INSERT INTO staff_presence_log (log_id, staff_id, log_date, present, reason)
VALUES (2, 3, '2025-07-25',
        0, 'Cover effort myself year body.');


INSERT INTO staff_presence_log (log_id, staff_id, log_date, present, reason)
VALUES (3, 1, '2025-07-03',
        1, 'Doctor know conference age plan civil.');


INSERT INTO staff_presence_log (log_id, staff_id, log_date, present, reason)
VALUES (4, 6, '2025-07-19',
        1, 'Black little suddenly machine present television sometimes card.');


INSERT INTO staff_presence_log (log_id, staff_id, log_date, present, reason)
VALUES (5, 6, '2025-07-06',
        0, 'Seem figure themselves conference.');


INSERT INTO staff_presence_log (log_id, staff_id, log_date, present, reason)
VALUES (6, 4, '2025-07-19',
        0, 'Recently return visit east.');


INSERT INTO staff_presence_log (log_id, staff_id, log_date, present, reason)
VALUES (7, 6, '2025-07-24',
        0, 'Popular energy town standard.');


INSERT INTO staff_presence_log (log_id, staff_id, log_date, present, reason)
VALUES (8, 9, '2025-07-01',
        1, 'Manager no decide customer season.');


INSERT INTO staff_presence_log (log_id, staff_id, log_date, present, reason)
VALUES (9, 9, '2025-06-30',
        0, 'Spend result ground run according.');


INSERT INTO staff_presence_log (log_id, staff_id, log_date, present, reason)
VALUES (10, 7, '2025-07-07',
        1, 'Fund to your method modern top.');

-- Staff Incidents Entries

INSERT INTO staff_incidents (incident_id, staff_id, incident_date, description, severity, action_taken)
VALUES (1, 4, '2025-05-18',
        'Thing happy school kind social agent camera.', 'Low', 'Whom where line theory join say.');


INSERT INTO staff_incidents (incident_id, staff_id, incident_date, description, severity, action_taken)
VALUES (2, 2, '2025-06-07',
        'Thus store herself difference order able.', 'Moderate', 'Especially common song manage knowledge.');


INSERT INTO staff_incidents (incident_id, staff_id, incident_date, description, severity, action_taken)
VALUES (3, 10, '2025-05-24',
        'Similar almost writer current song next bar.', 'Low', 'Subject room room money.');


INSERT INTO staff_incidents (incident_id, staff_id, incident_date, description, severity, action_taken)
VALUES (4, 5, '2025-07-05',
        'Bag test inside cup TV.', 'Low', 'Under some agree old make reflect long.');


INSERT INTO staff_incidents (incident_id, staff_id, incident_date, description, severity, action_taken)
VALUES (5, 4, '2025-06-01',
        'Kitchen sound six forget argue pick time.', 'Moderate', 'Become girl keep organization use officer only.');


INSERT INTO staff_incidents (incident_id, staff_id, incident_date, description, severity, action_taken)
VALUES (6, 6, '2025-04-29',
        'Same television language partner while lot vote.', 'Low', 'Try quickly choice doctor bill budget.');


INSERT INTO staff_incidents (incident_id, staff_id, incident_date, description, severity, action_taken)
VALUES (7, 10, '2025-05-21',
        'Week floor whether.', 'Low', 'Talk before with consider radio.');


INSERT INTO staff_incidents (incident_id, staff_id, incident_date, description, severity, action_taken)
VALUES (8, 2, '2025-06-28',
        'Kind people music poor purpose building positive.', 'Moderate', 'Mrs maintain would nearly type ok your.');


INSERT INTO staff_incidents (incident_id, staff_id, incident_date, description, severity, action_taken)
VALUES (9, 3, '2025-05-02',
        'Agent perhaps blood score become attention.', 'Moderate', 'Meet money eight big.');


INSERT INTO staff_incidents (incident_id, staff_id, incident_date, description, severity, action_taken)
VALUES (10, 9, '2025-07-21',
        'Full throw west of.', 'Low', 'Leave tonight board none pass.');

-- Sample room data (10 entries)
INSERT INTO rooms (room_id, room_number, room_type, is_available) VALUES (1, '101', 'Single', 1);
INSERT INTO rooms (room_id, room_number, room_type, is_available) VALUES (2, '102', 'Double', 1);
INSERT INTO rooms (room_id, room_number, room_type, is_available) VALUES (3, '103', 'Suite', 1);
INSERT INTO rooms (room_id, room_number, room_type, is_available) VALUES (4, '104', 'Deluxe', 1);
INSERT INTO rooms (room_id, room_number, room_type, is_available) VALUES (5, '105', 'Single', 1);
INSERT INTO rooms (room_id, room_number, room_type, is_available) VALUES (6, '106', 'Double', 0);
INSERT INTO rooms (room_id, room_number, room_type, is_available) VALUES (7, '107', 'Suite', 1);
INSERT INTO rooms (room_id, room_number, room_type, is_available) VALUES (8, '108', 'Single', 1);
INSERT INTO rooms (room_id, room_number, room_type, is_available) VALUES (9, '109', 'Deluxe', 0);
INSERT INTO rooms (room_id, room_number, room_type, is_available) VALUES (10, '110', 'Double', 1);