from src.Warehouse import *
from configs import *
from dataloader.Battery_Dataloader import battery_loader
from src.experiments import exper
from utils.tools import get_model_class
from peft import PeftModel

if __name__ == '__main__':
    model_names = ['PaiFilter','TimeMixer','DLinear','TSMixer','PatchTST', 'SegRNN', 'FiLM', 'FreTS', 'LightTS']
    for model_name in model_names:
        for cell in range(5, 9):

            print('start the experiment of {} on Cell{}'.format(model_name,cell))

            numepoch = 20
            embed = 'timeF'
            data_path = f'dataloader/dataset/CCCV/Cell{cell}.csv'
            model_path = 'teacher_model/model'
            cache_dir = './teacher_model'
            lora_path = 'teacher_model/lora_timer_battery'
            os.environ["TRANSFORMERS_CACHE"] = cache_dir

            seq_len, label_len, pred_len = 96,96,96
            freq = 'h'
            train_data, train_loader = battery_loader(embed,data_path,seq_len, label_len, pred_len, freq, flag='train')
            vali_data, vali_loader = battery_loader(embed,data_path,seq_len, label_len, pred_len, freq, flag='val')
            test_data, test_loader = battery_loader(embed,data_path,seq_len, label_len, pred_len, freq, flag='test')

            Model = get_model_class(model_name)
            base_model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, local_files_only=True, cache_dir=cache_dir)
            teacher_model = PeftModel.from_pretrained(base_model, lora_path, cache_dir=cache_dir)
            student_model = Model(args).float().to(device)
            model_optim = optim.Adam(student_model.parameters(), lr=1e-5)
            criterion = nn.MSELoss().to(device)

            try:

                exper(cell,model_name,numepoch,student_model,teacher_model,train_loader,model_optim,criterion,vali_data, vali_loader, test_data,test_loader, pred_len,label_len,device)

            except Exception as e:
                print(f"❌ {model_name} - {cell} failed: {e}")
