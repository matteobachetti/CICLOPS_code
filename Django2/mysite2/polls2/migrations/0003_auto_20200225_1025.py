# Generated by Django 3.0.2 on 2020-02-25 09:25

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('polls2', '0002_auto_20200213_1602'),
    ]

    operations = [
        migrations.CreateModel(
            name='PulseProfile',
            fields=[
                ('id', models.AutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pulse_freq', models.FloatField(default=0)),
                ('start_freq', models.IntegerField(default=0)),
                ('bandwidth', models.IntegerField(default=0)),
                ('nchan', models.IntegerField(default=0)),
                ('dm', models.FloatField(default=0)),
                ('amp', models.FloatField(default=0)),
                ('width', models.FloatField(default=0)),
                ('ph0', models.FloatField(default=0)),
                ('nbin', models.IntegerField(default=0)),
                ('noise_level', models.FloatField(default=0)),
                ('dedisp_profile', models.FileField(default=None, upload_to='')),
                ('allprofs', models.FileField(default=None, upload_to='')),
                ('disp_profile', models.FileField(default=None, upload_to='')),
                ('disp_z2', models.FloatField(default=0)),
                ('disp_z6', models.FloatField(default=0)),
                ('disp_z12', models.FloatField(default=0)),
                ('disp_z20', models.FloatField(default=0)),
                ('disp_H', models.FloatField(default=0)),
                ('dedisp_z2', models.FloatField(default=0)),
                ('dedisp_z6', models.FloatField(default=0)),
                ('dedisp_z12', models.FloatField(default=0)),
                ('dedisp_z20', models.FloatField(default=0)),
                ('dedisp_H', models.FloatField(default=0)),
                ('bin_dedisp_profile', models.FilePathField(path=None)),
                ('bin_allprofs', models.FilePathField(path=None)),
                ('bin_disp_profile', models.FilePathField(path=None)),
            ],
        ),
        migrations.DeleteModel(
            name='Database',
        ),
    ]
