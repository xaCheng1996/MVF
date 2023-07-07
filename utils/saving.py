import torch


def epoch_saving_voice_face(epoch, voice_model, face_model, optimizer, filename):
    torch.save({
                    'epoch': epoch,
                    'voice_model_state_dict': voice_model.state_dict(),
                    'face_model_state_dict': face_model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    }, filename)


def best_saving_voice_face(working_dir, epoch, voice_model, face_model, optimizer):
    best_name = '{}/model_best.pt'.format(working_dir)
    torch.save({
        'epoch': epoch,
        'voice_model_state_dict': voice_model.state_dict(),
        'face_model_state_dict': face_model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, best_name)