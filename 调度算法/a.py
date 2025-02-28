import heapq

# RM调度算法实现
def rm_schedule(tasks):
    # 按周期从小到大排序
    tasks = sorted([(tasks[i][1], i, tasks[i][0], tasks[i][1]) for i in range(len(tasks))])
    schedule = []
    time = 0
    preempted_times = 0
    task_queue = []
    task_instances = [(0, i, tasks[i][2], tasks[i][3]) for i in range(len(tasks))]

    while task_instances or task_queue:
        while task_instances and task_instances[0][0] <= time:
            task_instance = heapq.heappop(task_instances)
            heapq.heappush(task_queue, (task_instance[1], task_instance))
        
        if task_queue:
            _, (start_time, task_id, exec_time, period) = heapq.heappop(task_queue)
            if exec_time > 0:
                schedule.append({'time': time, 'task': task_id + 1, 'action': 0})
                time += exec_time
                schedule.append({'time': time, 'task': task_id + 1, 'action': 1})
            for i in range(len(tasks)):
                heapq.heappush(task_instances, (time + tasks[i][3], i, tasks[i][2], tasks[i][3]))
        else:
            time += 1
    
    return {'events': schedule, 'end_time': time, 'preempted_times': preempted_times}

# EDF调度算法实现
def edf_schedule(tasks):
    schedule = []
    time = 0
    preempted_times = 0
    task_queue = []
    task_instances = [(tasks[i][1], i, tasks[i][0], tasks[i][1]) for i in range(len(tasks))]

    while task_instances or task_queue:
        while task_instances and task_instances[0][0] <= time:
            task_instance = heapq.heappop(task_instances)
            heapq.heappush(task_queue, (task_instance[0], task_instance))
        
        if task_queue:
            _, (deadline, task_id, exec_time, period) = heapq.heappop(task_queue)
            if exec_time > 0:
                schedule.append({'time': time, 'task': task_id + 1, 'action': 0})
                time += exec_time
                schedule.append({'time': time, 'task': task_id + 1, 'action': 1})
            for i in range(len(tasks)):
                heapq.heappush(task_instances, (time + tasks[i][1], i, tasks[i][0], tasks[i][1]))
        else:
            time += 1
    
    return {'events': schedule, 'end_time': time, 'preempted_times': preempted_times}

# 选择调度算法
def schedule(tasks, strategy):
    if strategy == 0:
        return rm_schedule(tasks)
    else:
        return edf_schedule(tasks)

def main():
    import sys
    input = sys.stdin.read
    data = input().split()
    
    index = 0
    R = int(data[index])
    index += 1
    S = int(data[index])
    index += 1
    D = int(data[index])
    index += 1
    N = int(data[index])
    index += 1
    
    tasks = []
    for _ in range(N):
        Ti = int(data[index])
        Pi = int(data[index + 1])
        index += 2
        tasks.append((Ti, Pi))
    
    for run in range(R):
        schedule_result = schedule(tasks, S)
        print(f"RUN {run + 1}")
        for event in schedule_result['events']:
            print(f"{event['time']} [{event['task']} {event['action']}]")
        print(f"END {schedule_result['end_time']}")
        print(f"PREEMPTED {schedule_result['preempted_times']}")

if __name__ == "__main__":
    main()
