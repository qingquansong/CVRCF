import numpy as np


class DataSet(object):
    def __init__(self, filename, args, mark_u_time_end, mark_v_time_end):
        self.filename = filename
        self.f = open(self.filename, 'r')
        self.max_batch_size = args.max_batch_size
        self.time_interval = 604800 * args.time_interval  # day 86400, week 604800, month 30/31 days 2592000/2678400
        self.finish = 0
        self.num_of_u = args.num_of_u
        self.num_of_v = args.num_of_v
        self.gran_u = 604800 * args.gran_u
        self.gran_v = 604800 * args.gran_v
        self.mark_u_time_end = mark_u_time_end
        self.mark_v_time_end = mark_v_time_end

    def get_batch_data(self):
        all_data = []
        inputs_u_idx = []
        inputs_v_idx = []
        inputs_idx_pair = []

        last_pos = self.f.tell()
        line = self.f.readline()
        self.f.seek(last_pos)
        data = line.rstrip('\n').split('\t')
        start_time = float(data[3])
        end_time = start_time + self.time_interval

        for i in range(self.max_batch_size):
            last_pos = self.f.tell()
            line = self.f.readline()
            if line != '':
                data = line.rstrip('\n').split('\t')
                if float(data[3]) <= end_time:
                    all_data.append(data)
                    inputs_u_idx.append(int(data[0]))
                    inputs_v_idx.append(int(data[1]))
                    inputs_idx_pair.append([int(data[0]), int(data[1])])
                else:
                    self.f.seek(last_pos)
                    print("date cut")
                    break
            else:
                self.finish = 1
                self.f.close()
                self.f = open(self.filename, 'r')
                break

        inputs_u_idx = np.sort(list(set(inputs_u_idx)))-1
        inputs_v_idx = np.sort(list(set(inputs_v_idx)))-1

        all_data = np.asarray(all_data, dtype=np.float32)

        # Get mapped inputs_idx_pair
        inputs_idx_pair = np.asarray(inputs_idx_pair)
        siz = inputs_idx_pair.shape
        tmp = np.concatenate((inputs_idx_pair, np.array(range(siz[0]), ndmin=2).T
                              , np.zeros([siz[0], 2]), np.expand_dims(all_data[:, 2], 1)), axis=1)
        tmp = tmp[tmp[:, 0].argsort()]
        # Map user indices
        u_idx = -1
        mark = 0
        for i in range(siz[0]):
            if mark == int(tmp[i, 0]):
                tmp[i, 0] = u_idx
            else:
                mark = int(tmp[i, 0])
                u_idx += 1
                tmp[i, 0] = u_idx

        tmp = tmp[tmp[:, 1].argsort()]

        # Map user indices
        v_idx = -1
        mark = 0
        for i in range(siz[0]):
            if mark == int(tmp[i, 1]):
                tmp[i, 1] = v_idx
            else:
                mark = int(tmp[i, 1])
                v_idx += 1
                tmp[i, 1] = v_idx

        # Reorder to the original order
        tmp = tmp[tmp[:, 2].argsort()]

        u_time = np.zeros(inputs_u_idx.shape[0])-1
        v_time = np.zeros(inputs_v_idx.shape[0])-1
        mark_u_time = np.zeros(inputs_u_idx.shape[0])-self.gran_u-1
        mark_v_time = np.zeros(inputs_v_idx.shape[0])-self.gran_v-1
        mark_u_time_end = self.mark_u_time_end[inputs_u_idx]
        mark_v_time_end = self.mark_v_time_end[inputs_v_idx]

        sp_u_indices = []
        sp_v_indices = []
        sp_u_val = []
        sp_v_val = []
        sp_u_indices_res = []
        sp_v_indices_res = []
        sp_u_val_res = []
        sp_v_val_res = []
        for i in range(siz[0]):  # siz[0]
            if mark_u_time[int(tmp[i, 0])] + self.gran_u <= all_data[i, 3]:
                u_time[int(tmp[i, 0])] += 1
                mark_u_time[int(tmp[i, 0])] = all_data[i, 3]
                tmp[i, 3] = u_time[int(tmp[i, 0])]

                sp_u_indices.append([int(tmp[i, 0]), int(tmp[i, 3]), int(all_data[i, 1])-1])
                sp_u_val.append(all_data[i, 2])
                sp_u_indices_res.append([int(tmp[i, 0]), int(tmp[i, 3]), 0])
                sp_u_val_res.append(all_data[i, 4])

                if int(tmp[i, 3]) > 0:
                    sp_u_indices_res.append([int(tmp[i, 0]), int(tmp[i, 3])-1, 1])
                    sp_u_val_res.append(mark_u_time_end[int(tmp[i, 0])])
                mark_u_time_end[int(tmp[i, 0])] = all_data[i, 3]

                if all_data[i, 4] == 0:
                    sp_u_indices_res.append([int(tmp[i, 0]), int(tmp[i, 3]), 2])
                    sp_u_val_res.append(1)
            else:
                tmp[i, 3] = u_time[int(tmp[i, 0])]

                sp_u_indices.append([int(tmp[i, 0]), int(tmp[i, 3]), int(all_data[i, 1])-1])
                sp_u_val.append(all_data[i, 2])
                mark_u_time_end[int(tmp[i, 0])] = all_data[i, 3]

            if mark_v_time[int(tmp[i, 1])] + self.gran_v <= all_data[i, 3]:
                v_time[int(tmp[i, 1])] += 1
                mark_v_time[int(tmp[i, 1])] = all_data[i, 3]
                tmp[i, 4] = v_time[int(tmp[i, 1])]

                sp_v_indices.append([int(tmp[i, 1]), int(tmp[i, 4]), int(all_data[i, 0])-1])
                sp_v_val.append(all_data[i, 2])

                sp_v_indices_res.append([int(tmp[i, 1]), int(tmp[i, 4]), 0])
                sp_v_val_res.append(all_data[i, 5])

                if int(tmp[i, 4]) > 0:
                    sp_v_indices_res.append([int(tmp[i, 1]), int(tmp[i, 4])-1, 1])
                    sp_v_val_res.append(mark_v_time_end[int(tmp[i, 1])])
                mark_v_time_end[int(tmp[i, 1])] = all_data[i, 3]

                if all_data[i, 5] == 0:
                    sp_v_indices_res.append([int(tmp[i, 1]), int(tmp[i, 4]), 2])
                    sp_v_val_res.append(1)
            else:
                tmp[i, 4] = v_time[int(tmp[i, 1])]

                sp_v_indices.append([int(tmp[i, 1]), int(tmp[i, 4]), int(all_data[i, 0])-1])
                sp_v_val.append(all_data[i, 2])

                mark_v_time_end[int(tmp[i, 1])] = all_data[i, 3]

        u_mode = max(tmp[:, 3])+1
        v_mode = max(tmp[:, 4])+1

        for i in range(inputs_u_idx.shape[0]):
            sp_u_indices_res.append([i, u_time[i], 1])
            sp_u_val_res.append(mark_u_time_end[i])

        for i in range(inputs_v_idx.shape[0]):
            sp_v_indices_res.append([i, v_time[i], 1])
            sp_v_val_res.append(mark_v_time_end[i])

        self.mark_u_time_end[inputs_u_idx] = mark_u_time_end
        self.mark_v_time_end[inputs_v_idx] = mark_v_time_end

        sp_u_shape = np.array([inputs_u_idx.shape[0], u_mode, self.num_of_v])
        sp_v_shape = np.array([inputs_v_idx.shape[0], v_mode, self.num_of_u])
        sp_u_shape_res = np.array([inputs_u_idx.shape[0], u_mode, 3])
        sp_v_shape_res = np.array([inputs_v_idx.shape[0], v_mode, 3])

        sp_u_indices = np.asarray(sp_u_indices, dtype=np.int32)
        sp_u_val = np.asarray(sp_u_val)

        sp_u_indices_res = np.asarray(sp_u_indices_res, dtype=np.int32)
        sp_u_val_res = np.asarray(sp_u_val_res)

        sp_v_indices = np.asarray(sp_v_indices, dtype=np.int32)
        sp_v_val = np.asarray(sp_v_val)

        sp_v_indices_res = np.asarray(sp_v_indices_res, dtype=np.int32)
        sp_v_val_res = np.asarray(sp_v_val_res)

        return sp_u_indices, sp_u_shape, sp_u_val, sp_u_indices_res, sp_u_shape_res, sp_u_val_res,\
               sp_v_indices, sp_v_shape, sp_v_val, sp_v_indices_res, sp_v_shape_res, sp_v_val_res,\
               inputs_u_idx, inputs_v_idx, tmp[:, [0, 1, 3, 4, 5]], all_data
