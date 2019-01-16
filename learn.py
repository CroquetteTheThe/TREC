from rnn import *
from functions import *

# Now let's define learn(), which learn a RNN some data
def learn(rnn, data_loader, dev_loader, num_epochs=1, great_analysis=False):
    device = torch.device("cuda" if use_cuda else "cpu")
    if use_cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
        rnn.cuda(device)
    
    # Preparing
    rnn.train()
    losses_train = []
    losses_dev = []
    criterion = nn.CrossEntropyLoss()
    
    max_acc_dev = -1
    pos_best_rnn = 0
    save_state(rnn, "best_rnn")

    for epoch in range(num_epochs):
        total_correct = 0
        total_target = 0
        
        for batch_idx, (data, target) in enumerate(data_loader):
            #rnn.train()
            seeding_random()
            data, target = data.to(device), target.to(device)
            
            output = rnn(data)
            
            loss = criterion(output, target)
            rnn.optimizer.zero_grad()
            loss.backward()
            rnn.optimizer.step()
            
            # Get the Accuracy
            
            _, predicted = torch.max(output.data, dim=1)
            correct = (predicted == target).sum().item()
            total_correct += correct
            total_target += target.size(0)
            
            # Print the progress
            if batch_idx % 500 == 0 or batch_idx % 500 == 1 or batch_idx == len(data_loader)-1:
                print('\r Train Epoch: {}/{} [{}/{} ({:.0f}%)]\t Loss: {:.6f}\t Accuracy: {}'.format(
                    epoch+1,
                    num_epochs,
                    batch_idx * len(data), 
                    len(data_loader.dataset),
                    100. * batch_idx / len(data_loader), 
                    loss.data.item(),
                    (total_correct / total_target) * 100),
                    end='')
                losses_train.append(loss.data.item())
                if great_analysis:
                    dev_data, dev_target = next(iter(dev_loader))
                    dev_data, dev_target = dev_data.to(device), dev_target.to(device)
                    output = rnn(dev_data)
                    loss = criterion(output, dev_target)
                    losses_dev.append(loss.data.item())
                    
                    
        print()
        acc_dev = getEfficience(rnn, dev_loader)*100
        if acc_dev > max_acc_dev:
            max_acc_dev = acc_dev
            pos_best_rnn = epoch
            save_state(rnn, "best_rnn")
        
        print("Dev set: accuracy: " + str(acc_dev) + "% | max acc: " + str(max_acc_dev)+"%")
        print()
    
    rnn = load_state(rnn, "best_rnn")
    # Return losses list, you can print them later if you want
    return {"losses_train":losses_train, "losses_dev":losses_dev, "pos_best":pos_best_rnn+1}
